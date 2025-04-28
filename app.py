import io
import csv
import streamlit as st
import pandas as pd

class StockTradeAnalyzer:
    def __init__(self):
        """
        Initialize the StockTradeAnalyzer class
        """
        self.raw_df = None
        self.clean_df = None
        self.result_df = None

    def csv2df(self, uploaded_file):
        try:
            # 讀取內容並轉成 StringIO，模擬檔案物件
            content = uploaded_file.read().decode('big5')
            f = io.StringIO(content)

            # 用 csv.reader 讀取
            reader = csv.reader(f)
            rows = []
            for i, row in enumerate(reader):
                if i < 5:  # 只印前5行，Debug用
                    print(row)
                rows.append(row)

            # 轉成 DataFrame
            if rows:
                header = rows[2]  # 第3行當標題
                data = rows[3:]   # 第4行開始是資料
                df = pd.DataFrame(data, columns=header)
                print("DataFrame created!")
                return df
        except Exception as e:
            print(f"讀取檔案失敗: {e}")
            return None

    def df2clean(self, df):
        
        left_df = df.iloc[:, [1, 2, 3, 4]].copy()
        left_df.columns = ['券商', '價格', '買進股數', '賣出股數']
        right_df = df.iloc[:, [7, 8, 9, 10]].copy()
        right_df.columns = ['券商', '價格', '買進股數', '賣出股數']

        combined_df = pd.concat([left_df, right_df], ignore_index=True)
        combined_df['價格'] = pd.to_numeric(combined_df['價格'], errors='coerce')
        combined_df['買進股數'] = pd.to_numeric(combined_df['買進股數'], errors='coerce').fillna(0).astype(int)
        combined_df['賣出股數'] = pd.to_numeric(combined_df['賣出股數'], errors='coerce').fillna(0).astype(int)

        return combined_df.dropna()
    

    def df2calculate(self, df):
        results = {}
        
        # Calculate for each broker
        for broker in df['券商'].unique():
            broker_data = df[df['券商'] == broker]
            
            # Buy data calculations
            buy_data = broker_data[broker_data['買進股數'] > 0]
            total_buy_shares = buy_data['買進股數'].sum() if not buy_data.empty else 0
            if not buy_data.empty:
                total_buy_amount = (buy_data['價格'] * buy_data['買進股數']).sum()
                avg_buy_price = round(total_buy_amount / total_buy_shares, 2)
            else:
                total_buy_amount = 0
                avg_buy_price = 0
                
            # Sell data calculations
            sell_data = broker_data[broker_data['賣出股數'] > 0]
            total_sell_shares = sell_data['賣出股數'].sum() if not sell_data.empty else 0
            if not sell_data.empty:
                total_sell_amount = (sell_data['價格'] * sell_data['賣出股數']).sum()
                avg_sell_price = round(total_sell_amount / total_sell_shares, 2)
            else:
                total_sell_amount = 0
                avg_sell_price = 0
            
            # Calculate day trade volume
            day_trade_volume = min(total_buy_shares, total_sell_shares)
    
            # Calculate profit/loss per lot
            if day_trade_volume == 0:
                profit_loss = 0
            else:
                profit_loss = (avg_sell_price - avg_buy_price)*1000
                
            # Calculate net buy/sell amount
            net_shares = total_buy_shares - total_sell_shares
            if net_shares > 0:  # Net buy
                net_buy_amount = (net_shares * avg_buy_price)/10000 if avg_buy_price is not None else 0
                net_sell_amount = 0
            else:  # Net sell
                net_buy_amount = 0
                net_sell_amount = (abs(net_shares) * avg_sell_price)/10000 if avg_sell_price is not None else 0
            # today = datetime.now().date()
            # date = today.strftime('%Y-%m-%d')
            
            # Store results
            results[broker] = {
                '券商': broker,  # 加入券商欄位
                '總買進股數': total_buy_shares,
                '平均買進價格': avg_buy_price,
                '總賣出股數': total_sell_shares,
                '平均賣出價格': avg_sell_price,
                '當沖數量': day_trade_volume,
                '總買進金額': round(total_buy_amount, 2),
                '總賣出金額': round(total_sell_amount, 2),
                '買超股數': net_shares if net_shares > 0 else 0,
                '賣超股數': abs(net_shares) if net_shares < 0 else 0,
                '買超金額_萬': round(net_buy_amount, 2),
                '賣超金額_萬': round(net_sell_amount, 2),
                '盈虧_每張': round(profit_loss, 2)
            }
        
        # Convert to DataFrame and round values
        result_df = pd.DataFrame.from_dict(results, orient='index')
        result_df = result_df.round(2)
        result_df = result_df.reset_index(drop=True)
        return result_df


# 這三個是你自己寫好的 function
def top20_buy(df):
    # 這裡放你的邏輯
    df_buy_20 = df.sort_values(by='買超股數', ascending=False).iloc[0:20,:]
    df_buy_20_clean = df_buy_20[['券商','總買進股數','平均買進價格','總賣出股數','平均賣出價格','買超股數','買超金額_萬']]
    df_buy_20_clean = df_buy_20_clean.reset_index(drop=True)
    df_buy_20_clean.index = df_buy_20_clean.index + 1
    df_buy_20_clean.index.name = '名次'
    return df_buy_20_clean

def top20_sell(df):
    # 這裡放你的邏輯
    df_sell_20 = df.sort_values(by='賣超股數', ascending=False).iloc[0:20,:]
    df_sell_20_clean = df_sell_20[['券商','總買進股數','平均買進價格','總賣出股數','平均賣出價格','賣超股數','賣超金額_萬']]
    df_sell_20_clean = df_sell_20_clean.reset_index(drop=True)
    df_sell_20_clean.index = df_sell_20_clean.index + 1
    df_sell_20_clean.index.name = '名次'
    return df_sell_20_clean

def top20_intraday(df):
    # 這裡放你的邏輯
    df_day_20 = df.sort_values(by='當沖數量', ascending=False).iloc[0:20,:]
    df_day_20_clean = df_day_20[['券商','總買進股數','平均買進價格','總賣出股數','平均賣出價格','當沖數量','盈虧_每張']]
    df_day_20_clean = df_day_20_clean.reset_index(drop=True)
    df_day_20_clean.index = df_day_20_clean.index + 1
    df_day_20_clean.index.name = '名次'
    return df_day_20_clean




test = StockTradeAnalyzer()

# --- Streamlit 主程式 ---

# 🔵 CSS 調小字體
st.markdown("""
    <style>
    div[data-testid="stDataFrame"] div {
        font-size: 12px;
    }
    div[data-testid="stTable"] div {
        font-size: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 頁面標題 ---
st.title("買賣日報表彙總分析")

# --- 上傳CSV ---
uploaded_file = st.file_uploader("上傳CSV檔案", type=["csv"])

if uploaded_file is not None:
    df_raw = test.csv2df(uploaded_file)
    df2 = test.df2clean(df_raw)
    df = test.df2calculate(df2)
    
    if df is not None:
        st.success("檔案已整理完成！")

        # 🔵【新增】篩選功能
        st.subheader("完整資料篩選區")
        
        # 🔵 價格範圍選擇 + 手動輸入
        min_price = float(df['平均買進價格'].min())
        max_price = float(df['平均買進價格'].max())

        # 🔵 用 session state 控制價格同步
        if "price_min" not in st.session_state:
            st.session_state.price_min = min_price
        if "price_max" not in st.session_state:
            st.session_state.price_max = max_price

        # 🔵 拉條
        price_range = st.slider(
            "拖曳選擇平均買進價格範圍",
            min_value=min_price,
            max_value=max_price,
            value=(st.session_state.price_min, st.session_state.price_max),
            step=0.5,
            key="slider_price_range"
        )
        
        # 🔵 文字輸入
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.price_min = st.number_input(
                "輸入最小價格",
                min_value=min_price,
                max_value=st.session_state.price_max,
                value=st.session_state.price_min,
                step=0.5,
                format="%.2f"
            )
        with col2:
            st.session_state.price_max = st.number_input(
                "輸入最大價格",
                min_value=st.session_state.price_min,
                max_value=max_price,
                value=st.session_state.price_max,
                step=0.5,
                format="%.2f"
            )

        # 🔵 確保滑桿和輸入框同步
        price_range = (st.session_state.price_min, st.session_state.price_max)

        # 🔵 券商名稱選擇
        all_brokers = df['券商'].dropna().unique().tolist()
        selected_brokers = st.multiselect(
            "選擇券商（可複選，不選代表全部）",
            options=all_brokers
        )

        # 🔵 套用篩選條件
        df_filtered = df[
            (df['平均買進價格'] >= price_range[0]) &
            (df['平均買進價格'] <= price_range[1])
        ]
        if selected_brokers:
            df_filtered = df_filtered[df_filtered['券商'].isin(selected_brokers)]

        # 🔵 顯示可以互動排序的完整表格
        st.subheader("完整資料（可篩選、排序）")
        st.dataframe(df_filtered, use_container_width=True)

        st.divider()

        # --- 以下跟之前一樣，Top20報表＋下載功能 ---
        st.subheader("📈 買超前20名")
        df_buy = top20_buy(df)
        st.table(df_buy)

        csv_buy = df_buy.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="下載買超前20名 CSV",
            data=csv_buy,
            file_name='買超前20名.csv',
            mime='text/csv'
        )

        st.divider()

        st.subheader("📉 賣超前20名")
        df_sell = top20_sell(df)
        st.table(df_sell)

        csv_sell = df_sell.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="下載賣超前20名 CSV",
            data=csv_sell,
            file_name='賣超前20名.csv',
            mime='text/csv'
        )

        st.divider()

        st.subheader("⚡ 當沖前20名")
        df_intraday = top20_intraday(df)
        st.table(df_intraday)

        csv_intraday = df_intraday.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="下載當沖前20名 CSV",
            data=csv_intraday,
            file_name='當沖前20名.csv',
            mime='text/csv'
        )
