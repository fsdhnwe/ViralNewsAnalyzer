import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import jieba
import re
from datetime import datetime, timedelta
import logging
from sqlalchemy import create_engine
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection
)
import mysql.connector

class NewsClustering:
    def __init__(self, mysql_config, milvus_config):
        """
        初始化新聞聚類分析器
        
        Args:
            mysql_config: MySQL連接配置
            milvus_config: Milvus連接配置
        """
        self.mysql_config = mysql_config
        self.milvus_config = milvus_config
        self.collection_name = "news_posts"

        # 設置日誌
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)        # 初始化 Milvus 连接
        try:
            connections.connect(
                alias="default",
                host=milvus_config['host'],
                port=milvus_config['port']
            )
            self.logger.info("Successfully connected to Milvus")
        except Exception as e:
            self.logger.error(f"Failed to connect to Milvus: {e}")
            
        # 初始化 TF-IDF 向量化器
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # 需要中文停用詞
            ngram_range=(1, 2)
        )
        
        # 添加 PCA 用於降維，將 1000 維 TF-IDF 向量降到 200 維
        self.pca = PCA(n_components=200)
        self.pca_fitted = False
        
        # 跨批次聚類狀態追蹤
        # self.cluster_representatives = {} # {global_cluster_id: {'texts': [texts], 'vectors': [vectors], 'centroids': vector}} # Old, replaced
        # self.global_cluster_mapping = {} # {post_id: global_cluster_id} # Old, removed
        self.cluster_centroids = {}  # {global_cluster_id: {'centroid': np.array, 'count': int}}
        self.next_global_cluster_id = 0
        self.global_tfidf_fitted = False

        # 创建 Milvus collection (文章向量)
        self._create_collection()
        
        # 创建 Milvus collection (聚類中心向量)
        self.centroid_collection_name = "cluster_centroids"
        self._create_centroid_collection()
        
    def _create_collection(self):
        """
        创建 Milvus collection 如果不存在
        """
        if utility.has_collection(self.collection_name):
            return
            
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="content_vector", dtype=DataType.FLOAT_VECTOR, dim=200),
            FieldSchema(name="created_time", dtype=DataType.INT64),
            FieldSchema(name="batch_cluster", dtype=DataType.INT64),
            FieldSchema(name="is_early_report", dtype=DataType.BOOL),
            FieldSchema(name="is_viral_event", dtype=DataType.BOOL)
        ]
        
        schema = CollectionSchema(fields, "新聞文章向量集合")
        collection = Collection(self.collection_name, schema)
    
    def _create_centroid_collection(self):
        """
        创建存儲聚類中心點的 Milvus collection
        """
        if utility.has_collection(self.centroid_collection_name):
            return
            
        # 定义字段
        fields = [
            FieldSchema(name="cluster_id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="centroid_vector", dtype=DataType.FLOAT_VECTOR, dim=200),  # 降維後的維度
            FieldSchema(name="count", dtype=DataType.INT64)
        ]
        
        schema = CollectionSchema(fields, "聚類中心點向量集合")
        collection = Collection(self.centroid_collection_name, schema)
          # 建立索引 - 使用 HNSW 代替 IVF_FLAT 來提高性能
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {
                "M": 16,              # 每个节点的最大边数
                "efConstruction": 64  # 建索引时的搜索宽度
            }
        }
        collection.create_index(
            field_name="centroid_vector",
            index_params=index_params
        )
        self.logger.info(f"已創建聚類中心點向量集合: {self.centroid_collection_name}")

    def preprocess_text(self, text):
        """
        文本預處理
        """
        if pd.isna(text):
            return ""
        
        # 清理HTML標籤和特殊字符
        text = re.sub(r'<[^>]+>', '', str(text))
        text = re.sub(r'[^\w\s]', '', text)
        
        # 中文分詞
        words = jieba.cut(text)
        return ' '.join(words)
    
    def load_data_batch(self, batch_size=1000, offset=0):
        """
        分批從MySQL載入數據
        """
        try:
            # 建立 SQLAlchemy engine
            engine = create_engine(
                f"mysql+mysqlconnector://{self.mysql_config['user']}:{self.mysql_config['password']}@{self.mysql_config['host']}/{self.mysql_config['database']}?charset=utf8mb4"
            )
            query = """
            SELECT id, created_time, reaction_all, comment_count, 
                   share_count, page_category, page_name, content
            FROM news_posts 
            ORDER BY created_time 
            LIMIT %s OFFSET %s
            """
            # 用 params 傳遞 LIMIT/OFFSET 參數
            df = pd.read_sql(query, engine, params=(batch_size, offset))
            engine.dispose()

            # 數據預處理
            df['created_time'] = pd.to_datetime(pd.to_numeric(df['created_time'], errors='coerce'), unit='ms')
            df['processed_content'] = df['content'].apply(self.preprocess_text)

            return df

        except Exception as e:
            self.logger.error(f"數據載入錯誤: {e}")
            return pd.DataFrame()
    
    def calculate_similarity_matrix(self, texts):
        """
        計算文本相似度矩陣
        """
        # TF-IDF向量化
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # 計算cosine相似度
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        return similarity_matrix
    
    def cluster_similar_posts(self, df, similarity_threshold=0.8):
        """
        基於相似度聚類相似文章 (批次內聚類)
        輸出 'batch_cluster' 欄位
        """
        if df.empty or len(df) < 2:
            df['batch_cluster'] = range(len(df)) # Ensure column exists
            return df

        # 計算相似度矩陣
        # Note: calculate_similarity_matrix uses fit_transform, which is fine for batch-local
        similarity_matrix = self.calculate_similarity_matrix(df['processed_content'])

        # 將相似度矩陣轉換為距離矩陣
        distance_matrix = 1 - similarity_matrix
        distance_matrix = np.clip(distance_matrix, 0, 1)
        np.fill_diagonal(distance_matrix, 0)

        # 使用DBSCAN聚類
        clustering = DBSCAN(
            metric='precomputed',
            eps=1-similarity_threshold,
            min_samples=1 # Each post will be part of a cluster, even if a singleton
        )

        cluster_labels = clustering.fit_predict(distance_matrix)
        df['batch_cluster'] = cluster_labels # Changed from event_cluster to batch_cluster

        # 處理噪音點（標籤為-1, though min_samples=1 should prevent this with DBSCAN)
        # If min_samples > 1, noise handling is important.
        # For min_samples=1, all points are core points or border points of a cluster of size 1.
        # So, -1 labels shouldn't appear unless eps is very restrictive.
        # However, keeping the noise handling for robustness if min_samples changes.
        noise_mask = df['batch_cluster'] == -1
        if noise_mask.any():
            max_cluster = df['batch_cluster'][~noise_mask].max() if (~noise_mask).any() else -1
            num_noise_points = noise_mask.sum()
            df.loc[noise_mask, 'batch_cluster'] = range(
                max_cluster + 1, 
                max_cluster + 1 + num_noise_points
            )
        return df
    def identify_early_reports(self, df):
        """
        識別每個事件的早期報導（基於全局聚類）
        """
        result_dfs = []
        
        for cluster_id in df['global_cluster_id'].unique():
            cluster_df = df[df['global_cluster_id'] == cluster_id].copy()
            
            # 按時間排序
            cluster_df = cluster_df.sort_values('created_time')
            
            # 標記前兩篇為早期報導
            cluster_df['is_early_report'] = 0
            if len(cluster_df) >= 1:
                cluster_df.iloc[:min(2, len(cluster_df)), 
                              cluster_df.columns.get_loc('is_early_report')] = 1
            
            result_dfs.append(cluster_df)
        
        return pd.concat(result_dfs, ignore_index=True)
    def check_viral_event(self, df, days_window=7, min_media_count=10):
        """
        檢查是否為爆紅事件（7日內≥10家媒體跟進）
        基於全局聚類ID
        """
        result_dfs = []
        
        for cluster_id in df['global_cluster_id'].unique():
            cluster_df = df[df['global_cluster_id'] == cluster_id].copy()
            
            if len(cluster_df) == 0:
                continue
                
            # 找到第一篇報導的時間
            first_report_time = cluster_df['created_time'].min()
            end_time = first_report_time + timedelta(days=days_window)
            
            # 計算7日內的媒體數量
            within_window = cluster_df[
                (cluster_df['created_time'] >= first_report_time) & 
                (cluster_df['created_time'] <= end_time)
            ]
            
            unique_media_count = within_window['page_name'].nunique()
            is_viral = unique_media_count >= min_media_count
            
            cluster_df['is_viral_event'] = is_viral
            cluster_df['media_count_7days'] = unique_media_count
            
            result_dfs.append(cluster_df)
        
        return pd.concat(result_dfs, ignore_index=True) if result_dfs else pd.DataFrame()
    
    def save_to_milvus(self, df):
        """
        將聚類結果保存到Milvus
        """
        try:
            collection = Collection(self.collection_name)
            
            # 檢查是否已建立索引
            has_index = False
            for idx in collection.indexes:
                if idx.field_name == "content_vector":
                    has_index = True
                    break            # 將內容轉換為向量
            content_vectors = self.tfidf_vectorizer.transform(df['processed_content']).toarray()

            # 再做 PCA 降維
            if not self.pca_fitted:
                self.pca.fit(content_vectors)
                self.pca_fitted = True
            content_vectors_reduced = self.pca.transform(content_vectors)

            # 準備數據
            entities = [
                df['id'].tolist(),
                content_vectors_reduced.tolist(),  # 這裡改成降維後的向量
                df['created_time'].astype(np.int64).tolist(),
                df['global_cluster_id'].tolist(),
                df['is_early_report'].tolist(),
                df['is_viral_event'].tolist()
            ]

            # 插入數據
            collection.insert(entities)            # 建立索引（如果尚未建立）
            if not has_index:
                index_params = {
                    "metric_type": "L2",
                    "index_type": "HNSW",
                    "params": {
                        "M": 16,              # 每个节点的最大边数
                        "efConstruction": 200  # 建索引时的搜索宽度，对于更大的向量集合需要更大的值
                    }
                }
                collection.create_index(
                    field_name="content_vector",
                    index_params=index_params
                )

            # 載入集合以確保數據可見
            collection.load()
            collection.flush()
            self.logger.info("數據已保存到Milvus")

        except Exception as e:
            self.logger.error(f"Milvus保存錯誤: {e}")
            
    def search_similar_posts(self, query_vector, limit=10):
        """
        在Milvus中搜索相似的文章
        """
        try:
            collection = Collection(self.collection_name)
            collection.load()
              # 执行向量搜索 - 针对HNSW的参数优化
            search_params = {"metric_type": "L2", "params": {"ef": 64}}  # ef参数控制搜索精度与速度的平衡
            results = collection.search(
                data=[query_vector],
                anns_field="content_vector",
                param=search_params,
                limit=limit,
                expr=None
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Milvus搜索錯誤: {e}")
            return None
    
    def assign_to_global_clusters(self, df, similarity_threshold=0.8):
        """
        將新批次的文章分配到全局聚類中，確保跨批次一致性
        使用聚類中心點表示法，每個全局聚類僅由其中心點向量表示
        使用 PCA 降維和 Milvus 向量索引優化查詢效率
        """
        if df.empty:
            return df
            
        # 首次處理時，需要fit TF-IDF vectorizer
        if not self.global_tfidf_fitted:
            self.tfidf_vectorizer.fit(df['processed_content'])
            self.global_tfidf_fitted = True
            
        # 計算新批次文章的向量 (使用全局fitted的vectorizer)
        new_vectors = self.tfidf_vectorizer.transform(df['processed_content']).toarray()
        
        # 使用PCA降維
        if not self.pca_fitted:
            self.pca.fit(new_vectors)
            self.pca_fitted = True
            
        # 將向量降維到較低維度
        reduced_vectors = self.pca.transform(new_vectors)
        
        # 為每篇文章分配全局cluster ID
        global_cluster_ids = []
        
        # 檢查是否已有現有的聚類中心點
        centroid_collection = Collection(self.centroid_collection_name)
        has_centroids = centroid_collection.num_entities > 0
        
        if has_centroids:
            # 加載聚類中心點集合
            centroid_collection.load()
            
            # 對每個文章向量執行ANN搜索
            for idx, (_, post_row) in enumerate(df.iterrows()):
                post_vector_reduced = reduced_vectors[idx]
                post_vector_original = new_vectors[idx]
                best_cluster_id = None
                best_similarity = 0
                  # 使用向量索引找到最相似的前5個聚類中心點 - 針對HNSW的參數優化
                search_params = {"metric_type": "L2", "params": {"ef": 32}}  # ef參數控制搜索精度
                results = centroid_collection.search(
                    data=[post_vector_reduced],
                    anns_field="centroid_vector",
                    param=search_params,
                    limit=5,  # 只檢查最相似的5個聚類
                    expr=None
                )
                
                if results and len(results) > 0 and len(results[0]) > 0:
                    # 進一步計算與這些候選聚類的相似度（使用原始向量空間的相似度）
                    for hit in results[0]:
                        global_cluster_id = hit.id
                        # 從內存中的cluster_centroids獲取原始向量
                        if global_cluster_id in self.cluster_centroids:
                            centroid = self.cluster_centroids[global_cluster_id]['centroid']
                            similarity = cosine_similarity([post_vector_original], [centroid])[0][0]
                            
                            if similarity >= similarity_threshold and similarity > best_similarity:
                                best_similarity = similarity
                                best_cluster_id = global_cluster_id
                
                # 分配聚類ID並更新中心點
                if best_cluster_id is not None:
                    global_cluster_ids.append(best_cluster_id)
                    # 更新聚類中心點（增量方式）
                    old_centroid = self.cluster_centroids[best_cluster_id]['centroid']
                    old_centroid_reduced = self.pca.transform([old_centroid])[0]
                    old_count = self.cluster_centroids[best_cluster_id]['count']
                    new_count = old_count + 1
                    
                    # 更新中心點為加權平均
                    new_centroid = (old_centroid * old_count + post_vector_original) / new_count
                    new_centroid_reduced = (old_centroid_reduced * old_count + post_vector_reduced) / new_count
                    
                    self.cluster_centroids[best_cluster_id]['centroid'] = new_centroid
                    self.cluster_centroids[best_cluster_id]['count'] = new_count
                    
                    # 更新Milvus中的聚類中心點
                    centroid_collection.delete(expr=f"cluster_id == {best_cluster_id}")
                    centroid_collection.insert([
                        [best_cluster_id],
                        [new_centroid_reduced.tolist()],
                        [new_count]
                    ])
                else:
                    # 創建新的全局聚類
                    new_global_cluster_id = self.next_global_cluster_id
                    self.next_global_cluster_id += 1
                    global_cluster_ids.append(new_global_cluster_id)
                    
                    self.cluster_centroids[new_global_cluster_id] = {
                        'centroid': post_vector_original.copy(),
                        'count': 1
                    }
                    
                    # 將新聚類中心點添加到Milvus
                    centroid_collection.insert([
                        [new_global_cluster_id],
                        [post_vector_reduced.tolist()],
                        [1]
                    ])
        else:
            # 如果沒有現有的聚類中心點，為每篇文章創建新的聚類
            for idx, (_, post_row) in enumerate(df.iterrows()):
                post_vector_original = new_vectors[idx]
                post_vector_reduced = reduced_vectors[idx]
                
                new_global_cluster_id = self.next_global_cluster_id
                self.next_global_cluster_id += 1
                global_cluster_ids.append(new_global_cluster_id)
                
                self.cluster_centroids[new_global_cluster_id] = {
                    'centroid': post_vector_original.copy(),
                    'count': 1
                }
                
                # 將新聚類中心點添加到Milvus
                centroid_collection.insert([
                    [new_global_cluster_id],
                    [post_vector_reduced.tolist()],
                    [1]                ])
            centroid_collection.load()
        
        # 更新DataFrame，添加全局聚類ID列
        df['global_cluster_id'] = global_cluster_ids
            
        return df
    def process_batch(self, batch_size=1000, offset=0):
        """
        處理單個批次的數據
        """
        self.logger.info(f"處理批次: offset={offset}, batch_size={batch_size}")
        
        # 載入數據
        df = self.load_data_batch(batch_size, offset)
        if df.empty:
            return df
        
        # 聚類相似文章（批次內聚類）
        df = self.cluster_similar_posts(df)
        
        # 分配全局聚類ID（使用ANN向量索引優化的方法）
        df = self.assign_to_global_clusters(df)
        
        # 識別早期報導
        df = self.identify_early_reports(df)
        
        # 檢查爆紅事件
        df = self.check_viral_event(df)
        
        return df
    
    def run_full_analysis(self, batch_size=1000):
        """
        運行完整分析流程
        """
        offset = 0
        all_results = []
        
        while True:
            batch_df = self.process_batch(batch_size, offset)
            
            if batch_df.empty:
                break
                
            all_results.append(batch_df)
            
            # 保存當前批次到Milvus
            self.save_to_milvus(batch_df)
            
            offset += batch_size
            self.logger.info(f"完成批次 {offset//batch_size}")
        
        # 合併所有結果
        final_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        
        return final_df

# 使用示例
if __name__ == "__main__":
    # MySQL配置
    mysql_config = {
        'host': 'localhost',
        'user': 'root',
        'password': '12345678',
        'database': 'fbarticle'
    }
    
    # Milvus配置
    milvus_config = {
        'host': 'localhost',
        'port': '19530'
    }
    
    # 初始化並運行分析
    analyzer = NewsClustering(mysql_config, milvus_config)
    results = analyzer.run_full_analysis(batch_size=1000)
      # 輸出統計結果
    print(f"總共處理文章數: {len(results)}")
    print(f"識別事件群組數: {results['global_cluster_id'].nunique()}")
    print(f"早期報導數量: {results['is_early_report'].sum()}")
    print(f"爆紅事件數量: {results['is_viral_event'].sum()}")