import json
import mysql.connector
from mysql.connector import Error
import os
import glob

def import_json_to_mysql(json_file_path, db_config, table_name):
    """
    將 JSON 檔案的內容匯入到 MySQL 資料庫的指定表格中。

    Args:
        json_file_path (str): JSON 檔案的路徑。
        db_config (dict): MySQL 資料庫連線設定，包含 host, user, password, database。
        table_name (str): 要插入資料的資料表名稱。
    """
    try:
        # 讀取 JSON 檔案
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 檢查資料是否為列表
        if not isinstance(data, list):
            print("錯誤：JSON 檔案的根元素必須是一個列表。")
            return 0

        # 建立資料庫連線
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        if data:
            # 假設 JSON 列表中的每個物件都是一個字典，代表一行資料
            # 從第一個物件獲取欄位名稱
            columns = data[0].keys()
            columns_str = ', '.join([f"`{col}`" for col in columns]) # 使用反引號處理特殊欄位名稱
            placeholders = ', '.join(['%s'] * len(columns))
            
            insert_query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"

            success_count = 0
            for row in data:
                values = tuple(row.get(col) for col in columns) # 使用 .get() 避免 KeyError
                try:
                    cursor.execute(insert_query, values)
                    success_count += 1
                except Error as e:
                    print(f"插入資料時發生錯誤: {row} - {e}")
                    # 可以選擇在這裡回滾或繼續處理其他資料列
            
            conn.commit()
            print(f"成功將 {success_count} 筆資料匯入到資料表 {table_name}。")
            return success_count

        else:
            print("JSON 檔案中沒有資料可供匯入。")
            return 0

    except FileNotFoundError:
        print(f"錯誤：找不到 JSON 檔案 {json_file_path}")
    except Error as e:
        print(f"資料庫連線或操作時發生錯誤：{e}")
    except json.JSONDecodeError:
        print(f"錯誤：解析 JSON 檔案 {json_file_path} 失敗。請檢查檔案格式。")
    except Exception as e:
        print(f"發生未預期的錯誤：{e}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
            print("MySQL 連線已關閉。")
    
    return 0

def import_all_json_files(json_folder_path, db_config, table_name):
    """
    將資料夾中所有的 JSON 檔案匯入到 MySQL 資料庫。

    Args:
        json_folder_path (str): JSON 檔案資料夾的路徑。
        db_config (dict): MySQL 資料庫連線設定。
        table_name (str): 目標資料表名稱。
    """
    # 確保路徑結尾有斜線
    if not json_folder_path.endswith('\\') and not json_folder_path.endswith('/'):
        json_folder_path += '\\'
    
    # 取得所有 JSON 檔案
    json_files = glob.glob(json_folder_path + "*.json")
    
    if not json_files:
        print(f"錯誤：在 {json_folder_path} 中找不到任何 JSON 檔案。")
        return
    
    print(f"找到 {len(json_files)} 個 JSON 檔案需要處理。")
    
    total_records = 0
    processed_files = 0
    
    # 逐一處理每個 JSON 檔案
    for json_file in json_files:
        print(f"\n處理檔案: {os.path.basename(json_file)}")
        records_count = import_json_to_mysql(json_file, db_config, table_name)
        total_records += records_count
        processed_files += 1
    
    print("\n==== 匯入完成 ====")
    print(f"已處理 {processed_files} 個檔案")
    print(f"總共匯入 {total_records} 筆資料到資料表 {table_name}")

if __name__ == "__main__":
    # --- 請修改以下設定 ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 獲取當前腳本所在的目錄
    json_folder = os.path.join(BASE_DIR, "parsed_posts_jsonfile")  # JSON 檔案資料夾路徑
    
    # MySQL 資料庫連線設定
    db_connection_config = {
        'host': 'localhost',        # 資料庫主機名稱
        'user': 'root',             # 您的資料庫使用者名稱
        'password': '12345678',     # 您的資料庫密碼
        'database': 'fbarticle'     # 您的資料庫名稱
    }
    
    target_table = 'posts'          # 您要插入資料的資料表名稱
    # --- 設定結束 ---

    # 執行匯入
    import_all_json_files(json_folder, db_connection_config, target_table)