import mysql.connector

conn = mysql.connector.connect(
    host= 'localhost',
    user= 'root',
    password= '12345678',
    database= 'fbarticle'
)
cursor = conn.cursor()

batch_size = 500000
offset = 0

while True:
    sql = f"""
    INSERT INTO news_posts
    SELECT *
    FROM posts
    WHERE page_category IN (
      'Media/News Company', 'Journalist', 'News & Media Website',
      'Media', 'News Personality', 'Broadcasting & Media Production Company',
      'Publisher', 'TV Network', 'Radio Station', 'Newspaper', 'Media Agency',
      'TV Channel', 'Editorial/Opinion', 'Newsstand', 'Article', 'Editor', 'TV & Movies'
    )
    LIMIT {batch_size} OFFSET {offset}
    """
    cursor.execute(sql)
    conn.commit()
    
    # 如果一次搬不到 batch_size 筆，代表搬完了
    if cursor.rowcount < batch_size:
        break
    offset += batch_size

cursor.close()
conn.close()