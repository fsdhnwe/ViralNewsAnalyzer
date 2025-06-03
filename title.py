import csv
import re
import os
import io
import json
from datetime import datetime
import glob

# --- 配置開始 ---
# 定義輸入資料夾路徑和輸出資料夾路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_folder_path = os.path.join(BASE_DIR, "data")
output_folder_path = os.path.join(BASE_DIR, "parsed_posts_jsonfile")

# 確保輸出資料夾存在
os.makedirs(output_folder_path, exist_ok=True)

# 數據中欄位的順序 (0索引)
# 順序為: pos_tid, post_type, page_category, page_name, page_id, content, created_time, reaction_all, comment_count, share_count, date
DATA_FIELD_ORDER = [
    "pos_tid", "post_type", "page_category", "page_name", "page_id",
    "content", "created_time", "reaction_all", "comment_count", "share_count", "date"
]
EXPECTED_FIELD_COUNT_NORMAL = 11  # 標準記錄的欄位數
EXPECTED_FIELD_COUNT_SHORT = 6    # 短記錄（例如 ",num,num,num,num,date"）的欄位數

# 您最終希望在 JSON 中包含的欄位
DESIRED_JSON_FIELDS = [
    "created_time", "reaction_all", "comment_count", "share_count", "date",
    "post_type", "page_category", "page_name", "content"
]
# --- 配置結束 ---

def process_csv_file(input_file_path):
    """處理單個CSV檔案並生成對應的JSON檔案"""
    
    # 從檔名中提取日期（假設格式為 data_YYYY-M-D.csv）
    basename = os.path.basename(input_file_path)
    date_str = basename.replace("data_", "").replace(".csv", "")
    TARGET_DATE_SUFFIX = date_str
    
    # 設定輸出JSON檔案路徑
    output_file = os.path.join(output_folder_path, f"parsed_posts{date_str}.json")
    
    print(f"\n開始處理檔案: {basename}")
    print(f"日期識別為: {TARGET_DATE_SUFFIX}")
    
    all_extracted_records_for_json = []
    current_search_start_idx = 0 # 用於在原始數據字串中定位每條記錄的起始位置
    
    # 正規表示式，用於尋找記錄的結束標誌。
    record_terminator_pattern = re.compile(
        r"(?:,\s*\d+\s*){4}," + re.escape(TARGET_DATE_SUFFIX) + r"(?:\n|$)"
    )
    
    # 讀取檔案內容
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            input_file_content = file.read()
    except UnicodeDecodeError:
        # 如果UTF-8解碼失敗，嘗試其他編碼
        try:
            with open(input_file_path, 'r', encoding='cp950') as file:  # 嘗試使用繁體中文常用編碼
                input_file_content = file.read()
        except UnicodeDecodeError:
            print(f"錯誤：無法使用 UTF-8 或 CP950 解碼檔案 {input_file_path}")
            return

    # 遍歷數據，尋找每個記錄的結束點
    for match in record_terminator_pattern.finditer(input_file_content):
        record_end_idx = match.end()  # 記錄結束標誌在原始數據中的結束位置
        
        # 提取當前記錄的字串
        current_record_string = input_file_content[current_search_start_idx:record_end_idx].strip()

        if not current_record_string: # 如果是空字串，則跳過
            current_search_start_idx = record_end_idx
            continue

        # 使用 io.StringIO 將（可能包含多行的）記錄字串模擬成一個檔案對象
        record_file_simulator = io.StringIO(current_record_string)
        csv_parser = csv.reader(record_file_simulator)

        try:
            # 對於一個格式正確的記錄字串，csv_parser 應該只生成一個欄位列表
            parsed_fields_list = next(csv_parser)

            # 進行一個額外的檢查：確保解析出的最後一個欄位確實是目標日期
            if parsed_fields_list and parsed_fields_list[-1].strip() == TARGET_DATE_SUFFIX:
                
                if len(parsed_fields_list) == EXPECTED_FIELD_COUNT_NORMAL:
                    # 這是標準的11欄位記錄
                    # 將解析出的欄位值與預定義的欄位名順序對應起來，存為字典
                    raw_record_data_dict = dict(zip(DATA_FIELD_ORDER, parsed_fields_list))
                    
                    # 準備只包含所需欄位的字典，用於最終的JSON輸出
                    json_output_entry = {}
                    all_desired_fields_present = True
                    for key in DESIRED_JSON_FIELDS:
                        if key in raw_record_data_dict:
                            json_output_entry[key] = raw_record_data_dict[key]
                        else:
                            # 理論上，如果 DATA_FIELD_ORDER 和 DESIRED_JSON_FIELDS 都正確，不應發生
                            print(f"警告：所需欄位 '{key}' 未在解析記錄中找到。記錄: {current_record_string[:100]}...")
                            all_desired_fields_present = False
                            break # 放棄此條記錄
                    
                    if all_desired_fields_present:
                        all_extracted_records_for_json.append(json_output_entry)

                elif len(parsed_fields_list) == EXPECTED_FIELD_COUNT_SHORT and parsed_fields_list[0] == '':
                    # 處理特殊的短記錄情況，例如：",1611335969000,3,38,0,2021-1-23"
                    print(f"資訊：跳過短記錄 (首欄位為空, 共 {len(parsed_fields_list)} 個欄位): {current_record_string[:100]}...")
                
                else:
                    # 記錄以目標日期結尾，但不符合已知的欄位數量
                    print(f"資訊：跳過日期正確但欄位數異常 ({len(parsed_fields_list)}) 的記錄: {current_record_string[:100]}...")

        except StopIteration:
            # csv_parser 未能從 record_file_simulator 中讀取任何行
            print(f"警告：無法從以下片段解析任何CSV行: {current_record_string[:100]}...")
        except csv.Error as e:
            # CSV格式錯誤
            print(f"警告：CSV解析錯誤，片段: {current_record_string[:100]}... 錯誤: {e}")

        # 更新下一次搜索的起始位置
        current_search_start_idx = record_end_idx

    # 將收集到的數據寫入JSON檔案
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(all_extracted_records_for_json, outfile, ensure_ascii=False, indent=4)
        print(f"成功：數據已提取到 {output_file}")
        print(f"檔案 {basename} 總共提取記錄數: {len(all_extracted_records_for_json)}")
        return len(all_extracted_records_for_json)
    except IOError as e:
        print(f"錯誤：無法寫入檔案 {output_file}。錯誤: {e}")
        return 0
    except Exception as e:
        print(f"寫入JSON時發生未知錯誤: {e}")
        return 0

def main():
    # 獲取所有CSV檔案路徑
    csv_files = glob.glob(os.path.join(input_folder_path, "data_*.csv"))
    
    if not csv_files:
        print(f"在 {input_folder_path} 找不到符合 'data_*.csv' 格式的檔案。")
        return
    
    print(f"找到 {len(csv_files)} 個CSV檔案需要處理。")
    
    total_records = 0
    processed_files = 0
    
    # 處理每個檔案
    for csv_file in csv_files:
        records_count = process_csv_file(csv_file)
        total_records += records_count
        processed_files += 1
    
    print("\n==== 處理完成 ====")
    print(f"已處理 {processed_files} 個檔案")
    print(f"總共提取 {total_records} 條記錄")
    print(f"輸出的JSON檔案位於: {output_folder_path}")

if __name__ == "__main__":
    main()
