
######## ASSOCIATION RULE LEARNING RECOMMENDER #########

### İŞ PROBLEMİ ###

# Sepet aşamasındaki kullanıcılara ürün önerisinde bulunma


### VERİ SETİ HİKAYESİ ###

# Online Retail II isimli veri seti İngiltere merkezli online bir satış
# mağazasının 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.
# Bu şirketin ürün kataloğunda hediyelik eşyalar yer alıyor. Promosyon ürünleri olarak da düşünülebilir.
# Çoğu müşterisinin toptancı olduğu bilgisi de mevcut.

### DEĞİŞKENLER ###

# InvoiceNo – Fatura Numarası
# Eğer bu kod C ile başlıyorsa işlemin iptal edildiğini ifade eder.
# StockCode – Ürün kodu
# Her bir ürün için eşsiz numara.
# Description – Ürün ismi
# Quantity – Ürün adedi
# Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate – Fatura tarihi
# UnitPrice – Fatura fiyatı (Sterlin)
# CustomerID – Eşsiz müşteri numarası
# Country – Ülke ismi

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# GÖREV-1: VERİ ÖN İŞLEME

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.info()
df.head()

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)

# GÖREV-2 : Germany müşterileri üzerinden birliktelik kuralları üretiniz.

df_gr = df[df['Country'] == "Germany"]

df_gr.groupby(["Invoice","Description"]).agg({"Quantity":"sum"}).head(20)
df_gr.groupby(["Invoice","Description"]).agg({"Quantity":"sum"}).unstack().iloc[0:5,0:5]
df_gr.groupby(["Invoice","Description"]).agg({"Quantity":"sum"}).unstack().fillna(0).iloc[0:5,0:5]
df_gr.groupby(["Invoice","Description"]).agg({"Quantity":"sum"}).unstack().fillna(0).applymap(
    lambda x: 1 if x>0 else 0).iloc[0:5,0:5]

df_gr.groupby(['Invoice', 'Description']). \
    agg({"Quantity": "sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]

def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0).\
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(["Invoice","Description"])["Quantity"].sum().unstack().fillna(0).\
            applymap(lambda x:1 if x > 0 else 0)

gr_inv_pro_df=create_invoice_product_df(df_gr)

gr_inv_pro_df=create_invoice_product_df(df_gr, id=True)

# Ürünün stock code nun hangi ürüne karşılık geldiği.
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df_gr, 10002)

# Birliktelik kuralının çıkarılması

frequent_itemsets=apriori(gr_inv_pro_df, min_support=0.01, use_colnames=True)
rules= association_rules(frequent_itemsets, metric="support", min_threshold=0.01)

def create_rules(gr_inv_pro_df, id=True):
    frequent_itemsets=apriori(gr_inv_pro_df, min_support=0.01, use_colnames=True)
    rules=association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

create_rules(gr_inv_pro_df).head(20)

# GÖREV-3 :ID'leri verilen ürünlerin isimleri nelerdir?
# Kullanıcı 1 ürün id'si: 21987
# Kullanıcı 2 ürün id'si: 23235
# Kullanıcı 3 ürün id'si: 22747

product_id= 21987

check_id(df_gr, 21987)

check_id(df_gr, 23235)

check_id(df_gr, 22747)

#GÖREV-4: Sepetteki kullanıcılar için ürün önerisi yapınız.

sorted_rules= rules.sort_values("lift", ascending=False)

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})

    return recommendation_list[:rec_count]

arl_recommender(rules, 21987, 2)

arl_recommender(rules, 23235, 2)

arl_recommender(rules, 22747, 2)

# GÖREV-5 : Önerilen ürünlerin isimleri nelerdir?

# 21987 önerilen ürünler

check_id(df_gr, 21124)

check_id(df_gr, 23307)

# 23235 önerilen ürünler

check_id(df_gr, 20750)

check_id(df_gr, 22037)

# 22747 önerilen ürünler

check_id(df_gr, 20750 )

check_id(df_gr, 22423)