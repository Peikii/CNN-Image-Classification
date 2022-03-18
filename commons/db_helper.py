import pymssql

def get_conn(config):
    """
    获取连接
    :param config:
    :return:
    """
    try:
        conn = pymssql.connect(server=config['server'],
                               user=config['user'],
                               password=config['password'],
                               database=config['database'])

        return conn
    except Exception as e:
        raise e

def execute_select(config, sql):
    """
    查一条数据
    :param config:
    :param sql:
    :return:
    """
    try:
        conn = get_conn(config)
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            return ''
    except Exception as e:
        raise e
    finally:
        conn.close()

def execute_batch_select(config, sql):
    """
    批量查
    :param config:
    :param sql:
    :return:
    """
    try:
        conn = get_conn(config)
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()

        return result
    except Exception as e:
        raise e
    finally:
        conn.close()

def execute_insert(config, sql):
    """
    插入
    :param config:
    :param sql:
    :return: id
    """

    try:
        conn = get_conn(config)
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchone()
        conn.commit()
        # return result
        if result:
            return result[0]
        else:
            return None
    except Exception as e:
        raise e
    finally:
        conn.close()

def execute_update(config, sql):
    """
    更新
    :param config:
    :param sql:
    :return:
    """

    try:
        conn = get_conn(config)
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
    except Exception as e:
        raise e
    finally:
        conn.close()

if __name__ == '__main__':
    # import time
    # from configobj import ConfigObj
    # config = ConfigObj("../configs/config.ini", encoding='UTF8')
    # datas = execute_batch_select(config['ProductDatabase'],"""
    # select id, pdf_path
    # from RPADB.dbo.VAT_PDF_Invoice_Source_Data
    # where CHARINDEX('cnshf-fs02', pdf_path)>0 and vendor_name='Asheville THERMO FISHER SCIENTIFIC(US)' and creation_time>'2020-03-20 01:00:00' order by creation_time desc;
    # """)
    # for row in datas:
    #     pdf_path = row[1].replace("TPP", "CTC")
    #     # print(pdf_path)
    #     up = "UPDATE RPADB.dbo.VAT_PDF_Invoice_Source_Data SET pdf_path='{}' WHERE id = {}".format(pdf_path, row[0])
    #     print(up)
    #     execute_update(config['ProductDatabase'], up)


    # print(datas)
    # from configobj import ConfigObj
    # config = ConfigObj("../configs/config.ini", encoding='UTF8')
    # email_purchOrg = execute_batch_select(config['TestDatabase'], """
    #                                                                     select MD5Code,Sender,Recipient,id,PurchaseOrg from RPADB.dbo.VAT_Outlook_Atta;
    #                                                                     """)
    # print(email_purchOrg)

    # from configobj import ConfigObj
    # import time
    # config = ConfigObj("../configs/config.ini", encoding='UTF8')
    # files = execute_batch_select(config['ProductDatabase'], """
    #                                                     select id, pdf_path from RPADB.dbo.VAT_PDF_extract_model order by id desc;
    #                                                     """)
    # rt = r'\\cnshf-fs02\appshare\TFS_Automation_CN'
    # for file in files:
    #     id = file[0]
    #     path = file[1].replace('/', '\\')
    #     if path.find(rt) != -1:
    #         new_path = path.replace(rt, '')[1:]
    #         print(new_path)
    #         up = "UPDATE RPADB.dbo.VAT_PDF_extract_model SET pdf_path='{}' WHERE id={}".format(new_path, id)
    #         execute_update(config['ProductDatabase'], up)
    #         time.sleep(20/1000)
    #
    #     else:
    #         pass

    # print(files)

    pass

