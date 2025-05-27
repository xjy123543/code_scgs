# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 13:00:25 2024

@author: Administrator
"""

import pymysql 
 
 #shujuku
config = { 'user': 'root',  'password': 'root',  'host': 'localhost',  'database': 'search'}
  
# 连接到MySQL数据库 清空 
cnx = pymysql.connect(**config)  
cursor = cnx.cursor()  
import pymysql  

# SQL语句  
sql_truncat = "DELETE FROM video WHERE video = '%https://www.bilibili.com/cheese/play/ss%'"  
  
try:  
    # 连接数据库  
    connection = pymysql.connect(**config)  
      
    with connection.cursor() as cursor:  
        # 执行SQL语句  
        cursor.execute(sql_truncat)  
          
        # 提交事务  
        connection.commit()  
          
        print("表数据已成功清空。")  
  
except pymysql.MySQLError as e:  
    print(f"数据库错误：{e}")  
  
finally:  
    # 关闭数据库连接  
    connection.close()