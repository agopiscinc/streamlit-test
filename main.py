# import  time
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from PIL import Image
import geopandas as gpd
import plotly.express as px
import json

# st.set_page_config(layout="wide")

st.title('GNS2021 市町村版')

st.write('試験公開のため，数値はすべて乱数によって生成したダミー値です。')


param_dict = {'GNS':'GNS', '人口':'P_NUM', '世帯数':'H_NUM', '曝露量':'曝露量', '地震':'地震', '津波':'津波', '高潮':'高潮', '土砂災害':'土砂災害', '土石流':'土石流', '急傾斜地':'急傾斜地', '地すべり':'地すべり', '火山':'火山', '洪水':'洪水',
       '脆弱性':'脆弱性', 'ハード':'ハード', '住宅・公共':'住宅・公共', 'ライフライン':'ライフライン', 'インフラ':'インフラ', '情報通信':'情報通信', 'ソフト':'ソフト',
       '物資・設備':'物資・設備', '医療サービス':'医療サービス', '経済と人口構成':'経済と人口構成', '保険':'保険', '条例自治':'条例自治'}
option_dict = st.sidebar.selectbox('コロプレス図の指標', param_dict.keys())


@st.cache(allow_output_mutation=True)
def load_data():
    # df = gpd.read_file('japan_ver821/japan_ver821.shp')
    df = pd.read_csv('input/df_dummy.csv')
    return df
df_japan = load_data()

# geometryがNoneの行を削除
df_japan = df_japan.dropna(subset=['geometry']).copy()
df_japan.reset_index(drop=True, inplace=True)


view_columns = ['names','JCODE','P_NUM', 'GNS','曝露量', '地震', '津波', '高潮', '土砂災害', '土石流', '急傾斜地', '地すべり', '火山', '洪水',
       '脆弱性', 'ハード', '住宅・公共', 'ライフライン', 'インフラ', '情報通信', 'ソフト',
       '物資・設備', '医療サービス', '経済と人口構成', '保険', '条例自治']
st.dataframe(df_japan[view_columns], width=1200, height=200)

# df_japan['name'] = df_japan['KEN'].astype(str) + df_japan['SEIREI'].astype(str) + df_japan['SIKUCHOSON'].astype(str)
# df_japan['name'] = df_japan['name'].replace('(.*)None(.*)', r'\1\2', regex=True)


#geometryの間引き
geometry = gpd.GeoSeries.from_wkt(df_japan['geometry'])
geometry = geometry.simplify(tolerance=0.001)


@st.cache(allow_output_mutation=True)
def geojson():
    return json.loads(geometry.to_json())
geojson = geojson()

# コロプレス図の中心 緯度，経度
CENTER = {'lat': 36, 'lon': 140}

@st.cache(allow_output_mutation=True)
def plot_choropleth(param):
    column = param_dict[param]
    fig = px.choropleth_mapbox(df_japan,
                               geojson=geojson,
                               locations=df_japan.index, #GeoJSONのIDに対応する項目
                               color=column,
                               width=800, height=800,
                               title = param,
                               hover_name = 'names',
                               color_continuous_scale='jet', #"OrRd",
                               mapbox_style='white-bg',
                               zoom=4,
                               center=CENTER,
                               opacity=1
                               # labels={'DENSITY':'人口密度'}
                              )
    fig.update_traces(marker_line_width=0.05)
    fig.update_layout(margin={'r':10,'t':40,'l':10,'b':10,'pad':5})
    return fig

fig = plot_choropleth(option_dict)


# left_col, right_col = st.columns(2)

# if st.checkbox('GNS choropleth'):
st.plotly_chart(fig)



# # 市町村の選択（それぞれ版）
# left_col, center_col, right_col = st.columns(3)
# option_ken = left_col.selectbox('都道府県',df_japan['KEN'].unique())
# option_seirei = center_col.selectbox('政令指定都市',df_japan[df_japan['KEN']==option_ken]['SEIREI'].unique())
# option_sikuchoson = right_col.selectbox('市区町村',df_japan[df_japan['KEN']==option_ken]['SIKUCHOSON'].unique())
# name = df_japan[(df_japan['KEN']==option_ken)&(df_japan['SIKUCHOSON']==option_sikuchoson)]['name'].values[0]

# st.text(f'選んだ都道府県は{option_ken}{option_seirei}{option_sikuchoson}です')

jcode_names = df_japan['names'].unique()
default_ix1 = int(np.where(jcode_names=='東京都練馬区')[0][0])
default_ix2 = int(np.where(jcode_names=='愛知県名古屋市中区')[0][0])
default_ix3 = int(np.where(jcode_names=='福岡県福岡市西区')[0][0])

st.sidebar.write('脆弱性指標')
names1 = st.sidebar.selectbox('市町村1', jcode_names, index=default_ix1)
names2 = st.sidebar.selectbox('市町村2', jcode_names, index=default_ix2)
names3 = st.sidebar.selectbox('市町村3', jcode_names, index=default_ix3)


# 鶏頭図


def coxcomb(col):
    # vulnerability = ['ハード', '住宅・公共', 'ライフライン', 'インフラ', '情報通信', 'ソフト', '物資・設備', '医療サービス', '経済と人口構成', '保険', '条例自治']
    vulnerability = ['住宅・公共', 'ライフライン', 'インフラ', '情報通信', '物資・設備', '医療サービス', '経済と人口構成', '保険', '条例自治']
    # df_plt = df_japan[df_japan['SIKUCHOSON']==option_sikuchoson]
    df_plt = df_japan[df_japan['names']==col]
    N = len(vulnerability)
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    # radii = 10 * np.random.rand(N)
    radii = df_plt[vulnerability].values[0]#/1000
    width = 2*np.pi/N
    # colors = plt.cm.viridis(radii / N)
    # colors= ['maroon','red','orange','gold','yellowgreen','green','aqua','blue', 'darkblue','purple','pink']
    colors= ['red','orange','gold','yellowgreen','green','aqua','blue', 'purple','pink']

    fig = plt.figure()#figsize=(4,4)
    ax = fig.add_subplot(111,projection='polar')
    ax.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=0.9)
    # ax.set_xticks(a,labels=None)
    # ax.axis("off")
    # ax.axes.xaxis.set_visible(False)
    # ax.axes.yaxis.set_visible(False)
    return fig



# N = len(vulnerability)
# theta = np.linspace(0.0, 2*np.pi, N, endpoint=False)
# # radii = 10 * np.random.rand(N)
# radii = df[vulnerability].values[1]#/1000
# width = 2*np.pi/(N)
# # colors = plt.cm.viridis(theta / N)
# colors= ['maroon','red','orange','gold','yellowgreen','green','aqua','blue', 'darkblue','purple','pink']

# ax = plt.subplot(projection='polar')
# ax.bar(theta, radii, width=width, bottom=0.0, color=colors, alpha=0.9)





left_col, center_col, right_col = st.columns(3)

left_col.pyplot(coxcomb(names1))
# left_col.text(f"{names1}")
left_col.markdown(f"<p style='text-align: center;'>{names1}</p>", unsafe_allow_html=True)

center_col.pyplot(coxcomb(names2))
# center_col.text(f"{names2}")
center_col.markdown(f"<p style='text-align: center;'>{names2}</p>", unsafe_allow_html=True)

right_col.pyplot(coxcomb(names3))
# right_col.text(f"{names3}")
right_col.markdown(f"<p style='text-align: center;'>{names3}</p>", unsafe_allow_html=True)






