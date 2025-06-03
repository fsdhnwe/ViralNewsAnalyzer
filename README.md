# Find-breaking-news-tweets

## 步驟說明

### 1. 建立資料夾與下載資料
- 在專案根目錄下建立 `data` 與 `parsed_posts_jsonfile` 兩個資料夾。
- 前往 Google 雲端硬碟下載原始資料（csv 檔），放入 `data` 資料夾中。

### 2. 解析原始資料
- 執行 `title.py`：
  ```bash
  python title.py
  ```
- 執行完成後，會在 `parsed_posts_jsonfile` 資料夾中看到每一天的 json 檔案。

### 3. 匯入 MySQL
- 下載並安裝 MySQL。
- 建立資料庫與 `posts` 資料表，請執行下方 SQL 建立 table：

  ```sql
  CREATE TABLE posts (
    id INT PRIMARY KEY AUTO_INCREMENT,
    created_time VARCHAR(20),
    reaction_all INT,
    comment_count INT,
    share_count INT,
    date DATE,
    post_type VARCHAR(255),
    page_category VARCHAR(255),
    page_name VARCHAR(255),
    content TEXT,
    label VARCHAR(30)
  );
  ```

- 請記得修改 `import_json_to_mysql.py` 中的 MySQL 連線設定（host、user、password、database）。
- 執行 `import_json_to_mysql.py` 匯入資料，總筆數約 1900 萬。
- 建議安裝 SQL 可視化工具（如 DBeaver、HeidiSQL、phpMyAdmin）輔助查詢。

### 4. 提取新聞貼文
- 執行 `auto_news_posts_batch.py`，會從所有資料中提取出只有 news 的 posts。
- 會自動建立一個新 table 叫 `news_posts`，其結構與 `posts` 相同，請執行下方 SQL 建立 table：

  ```sql
  CREATE TABLE news_posts (
    id INT PRIMARY KEY AUTO_INCREMENT,
    created_time VARCHAR(20),
    reaction_all INT,
    comment_count INT,
    share_count INT,
    date DATE,
    post_type VARCHAR(255),
    page_category VARCHAR(255),
    page_name VARCHAR(255),
    content TEXT,
    label VARCHAR(30)
  );
  ```

- 新聞 posts 約 190 萬筆，若刪除 reaction_all = 0 的資料，約剩 150 萬筆。

---

如有問題歡迎提問！