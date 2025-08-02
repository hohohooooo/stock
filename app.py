import io
import csv
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import io
from datetime import datetime
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

# today = datetime.now().date()
# date = today.strftime('%Y-%m-%d')

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
                '買入': total_buy_shares,
                '買入價': round(avg_buy_price,2),
                '賣出': total_sell_shares,
                '賣出價': round(avg_sell_price,2),
                '當沖量': day_trade_volume,
                '總買進金額': round(total_buy_amount, 2),
                '總賣出金額': round(total_sell_amount, 2),
                '淨買入': net_shares if net_shares > 0 else 0,
                '淨賣出': abs(net_shares) if net_shares < 0 else 0,
                '淨買額': int(net_buy_amount),
                '淨賣額': int(net_sell_amount),
                '當沖盈虧': int(profit_loss)
            }
        
        # Convert to DataFrame and round values
        result_df = pd.DataFrame.from_dict(results, orient='index')
        result_df = result_df.round(2)
        result_df = result_df.reset_index(drop=True)
        
        return result_df


# 這三個是你自己寫好的 function
def top20_buy(df):
    # 這裡放你的邏輯
    df_buy_20 = df.sort_values(by='淨買入', ascending=False).iloc[0:20,:]
    df_buy_20_clean = df_buy_20[['券商','買入','買入價','賣出','賣出價','淨買入','淨買額']]
    df_buy_20_clean = df_buy_20_clean.reset_index(drop=True)
    df_buy_20_clean.index = df_buy_20_clean.index + 1
    df_buy_20_clean.index.name = '名次'
    df_buy_20_clean['買入價'] = df_buy_20_clean['買入價'].round(2)
    df_buy_20_clean['賣出價'] = df_buy_20_clean['賣出價'].round(2)

    return df_buy_20_clean

def top20_sell(df):
    # 這裡放你的邏輯
    df_sell_20 = df.sort_values(by='淨賣出', ascending=False).iloc[0:20,:]
    df_sell_20_clean = df_sell_20[['券商','買入','買入價','賣出','賣出價','淨賣出','淨賣額']]
    df_sell_20_clean = df_sell_20_clean.reset_index(drop=True)
    df_sell_20_clean.index = df_sell_20_clean.index + 1
    df_sell_20_clean.index.name = '名次'
    df_sell_20_clean = df_sell_20_clean.round(2)
    df_sell_20_clean['買入價'] = df_sell_20_clean['買入價'].round(2)
    df_sell_20_clean['賣出價'] = df_sell_20_clean['賣出價'].round(2)

    return df_sell_20_clean

def top20_intraday(df):
    # 這裡放你的邏輯
    df_day_20 = df.sort_values(by='當沖量', ascending=False).iloc[0:20,:]
    df_day_20_clean = df_day_20[['券商','買入','買入價','賣出','賣出價','當沖量','當沖盈虧']]
    df_day_20_clean = df_day_20_clean.reset_index(drop=True)
    df_day_20_clean.index = df_day_20_clean.index + 1
    df_day_20_clean.index.name = '名次'
    df_day_20_clean['買入價'] = df_day_20_clean['買入價'].round(2)
    df_day_20_clean['賣出價'] = df_day_20_clean['賣出價'].round(2)

    return df_day_20_clean

def parse_formatted_number(value):
    if pd.isna(value): return 0.0
    if isinstance(value, (int, float, np.number)): return float(value)
    if isinstance(value, str):
        value = value.strip().replace(',', '')
        if value.startswith('(') and value.endswith(')'):
            try: return -float(value[1:-1])
            except ValueError: return 0.0
        if '張' in value and '(' in value and ')' in value:
            try: num_part = value.split('張')[0]; return float(num_part) * 1000
            except ValueError: return 0.0
        try: return float(value)
        except ValueError: return 0.0
    return 0.0

def format_broker_name(name):
    if not isinstance(name, str): return ""
    return ''.join(char for char in name if '\u4e00' <= char <= '\u9fff' or '\u3000' <= char <= '\u303f' or '\uff00' <= char <= '\uffef' or char in '()-')

def format_volume_int(value):
    numeric_value = parse_formatted_number(value)
    if pd.isna(numeric_value) or numeric_value == 0: return "0"
    volume_in_k = numeric_value / 1000.0
    return f"{int(round(volume_in_k))}"

def format_volume_with_price_label(volume_val, price_val):
    """
    生成圖表中間列的價格和數量文本（分離）。
    返回: (price_text, volume_text)
    如果價格無效，price_text 為空字符串。
    """
    volume_text = format_volume_int(volume_val)
    if pd.isna(price_val) or price_val <= 0:
        price_text = ""
    else:
        price_text = f"({price_val:.1f})"
    return price_text, volume_text

def df_to_png_bytes(df, title,date):

    df.insert(0, '名次', range(1, len(df) + 1))
    # 設定繁體中文字體
    font_path = Path("fonts/NotoSansCJKtc-Regular.otf")
    prop = fm.FontProperties(fname=font_path)

    def adjust_column_widths(table, df, total_width=1.0):
        col_widths = []
        for col in df.columns:
            max_len = max(
                [len(str(col))] + [len(str(v)) for v in df[col].values]
            )
            col_widths.append(max_len)

        col_widths = np.array(col_widths, dtype=float)
        col_widths /= col_widths.sum()  # 正規化
        col_widths *= total_width       # 總寬度 = 1.0

        for col_idx, width in enumerate(col_widths):
            for row_idx in range(len(df) + 1):  # +1 包含表頭
                cell = table[(row_idx, col_idx)]
                cell.set_width(width)

    # 計算合理高度
    fig, ax = plt.subplots(figsize=(len(df.columns) * 1.8, len(df) * 0.48))
    ax.axis('off')

    # 左上標題
    ax.text(
        0.01, 0.98, title,
        fontsize=16, fontproperties=prop, color='#333333',
        ha='left', va='top', transform=ax.transAxes
    )

    # 右上日期
    ax.text(
        0.99, 0.05, date,
        fontsize=12, fontproperties=prop, color='#666666',
        ha='right', va='top', transform=ax.transAxes
    )

    # 建立 table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center'
    )

    # 配色
    header_text_color = '#FFFFFF'
    header_bg_color   = '#4A6FA5'
    text_color        = '#333333'
    even_row_color    = '#F7F7F7'
    odd_row_color     = '#FFFFFF'
    edge_color        = '#DDDDDD'

    # 格式化 cell
    for (row, col), cell in table.get_celld().items():
        cell.get_text().set_fontproperties(prop)
        cell.get_text().set_fontsize(10)
        cell.get_text().set_weight('normal')

        if row == 0:
            # 表頭
            cell.set_facecolor(header_bg_color)
            cell.get_text().set_color(header_text_color)
        else:
            cell.get_text().set_color(text_color)
            if row % 2 == 0:
                cell.set_facecolor(even_row_color)
            else:
                cell.set_facecolor(odd_row_color)

        cell.set_edgecolor(edge_color)

    table.scale(1, 1.8)

    # 欄寬自適應
    adjust_column_widths(table, df)

    # 輸出
    buf = io.BytesIO()
    plt.savefig(
        buf,
        format="png",
        bbox_inches='tight',
        dpi=150,
        facecolor='white',
        pad_inches=0.02
    )
    buf.seek(0)
    plt.close(fig)
    return buf

def create_visualization(buy_top_raw, sell_top_raw, date, output_file="stock_analysis_visualization_final.png"):
    """生成表格樣式佈局，增大字體並調整佈局以適應 (v23)"""
    print("DEBUG: Entering create_visualization (v23 - larger fonts)...") # Version Update
    # 設定字型路徑
    # https://github.com/notofonts/noto-cjk/tree/main/Sans/OTF/TraditionalChinese
    font_path = Path("fonts/NotoSansCJKtc-Bold.otf")
    if font_path.exists():
        font_prop = fm.FontProperties(fname=str(font_path))
        plt.rcParams['font.family'] = font_prop.get_name()
        print("✅ 成功載入字型：", font_prop.get_name())
    else:
        print("❌ 找不到字型檔案，請檢查路徑")
    # plt.rcParams['font.family'] = 'sans-serif'
    

    if buy_top_raw.empty and sell_top_raw.empty:
        print("ERROR: Both buy and sell data are empty, cannot create visualization.")
        return

    n_items = 20
    num_buy = min(len(buy_top_raw), n_items)
    num_sell = min(len(sell_top_raw), n_items)
    max_rows = max(num_buy, num_sell)
    print(f"DEBUG: Plotting: buys={num_buy}, sells={num_sell}, max_rows={max_rows}")

    # --- Colors and Fonts ---
    bg_color = '#33373D'; header_color = '#AEAEAE'; broker_color = '#FFFFFF'
    buy_color_bar = '#A02C2C'; sell_color_bar = '#2E7D32'
    buy_price_color = '#E57373'; sell_price_color = '#66BB6A'
    buy_volume_color = '#FF7A7A'; sell_volume_color = '#81C784';
    summary_color = '#E0E0E0'

    # 字體大小 (v23)
    header_fontsize = 19
    broker_fontsize = 18
    value_fontsize = 19
    summary_fontsize = 17
    font_weight = 'bold'

    # --- Create Figure ---
    # 佈局調整 (v23)
    row_height_factor = 0.65
    fig_height = 1.8 + max_rows * row_height_factor + 0.8
    fig, ax = plt.subplots(figsize=(9, fig_height), facecolor=bg_color)
    fig.patch.set_facecolor(bg_color); ax.set_facecolor(bg_color)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_ylim(max_rows + 1.2, -1.5); ax.set_xlim(-0.05, 1.05)

    # --- Draw Header ---
    header_y = -0.5
    ax.text(0.5, -1.2, date, color=header_color, fontsize=header_fontsize, fontweight=font_weight, ha='center', va='center', fontproperties=font_prop)
    ax.text(0.18, header_y, "買超分點", color=header_color, fontsize=header_fontsize, fontweight=font_weight, ha='center', va='center', fontproperties=font_prop)
    ax.text(0.5, header_y, "買賣超張數(價)", color=header_color, fontsize=header_fontsize, fontweight=font_weight, ha='center', va='center', fontproperties=font_prop)
    ax.text(0.82, header_y, "賣超分點", color=header_color, fontsize=header_fontsize, fontweight=font_weight, ha='center', va='center', fontproperties=font_prop)

    # --- Prepare Data and Max Value (無變更) ---
    all_volumes_k = [];
    if num_buy > 0: all_volumes_k.extend(abs(buy_top_raw['淨買入'].head(num_buy) / 1000))
    if num_sell > 0: all_volumes_k.extend(abs(sell_top_raw['淨賣出'].head(num_sell) / 1000))
    if not all_volumes_k:
        max_abs_volume_k = 1.0
    else:
        max_abs_volume_k = max(all_volumes_k) if max(all_volumes_k) > 0 else 1.0
    print(f"DEBUG: Max absolute volume (K shares): {max_abs_volume_k}")

    # --- Draw Data Rows (v23 - Larger Fonts, Adjusted Layout) ---
    x_buy_broker = 0.01; x_sell_broker = 0.99
    x_volume_center = 0.5
    center_ref = x_volume_center
    base_offset = 0.015

    # 佈局參數 (v23)
    fixed_gap_value = 0.08
    bar_max_relative_width = 0.40
    bar_height = 0.7

    print(f"DEBUG: Drawing data rows with larger fonts (H:{header_fontsize}, B:{broker_fontsize}, V:{value_fontsize}, S:{summary_fontsize}) and adjusted layout (row_h:{row_height_factor}, bar_h:{bar_height})...")
    for i in range(max_rows):
        y = i + 0.5
        # Buy side
        if i < num_buy:
            try:
                broker = format_broker_name(buy_top_raw['券商'].iloc[i])
                volume_val = buy_top_raw['淨買入'].iloc[i]
                price_val = buy_top_raw['買入價'].iloc[i]
                price_text, volume_text = format_volume_with_price_label(volume_val, price_val)

                volume_k = abs(volume_val / 1000)
                bar_width = (volume_k / max_abs_volume_k) * bar_max_relative_width if max_abs_volume_k > 0 else 0
                bar_left = center_ref - bar_width

                ax.barh(y, width=bar_width, left=bar_left, height=bar_height, color=buy_color_bar, alpha=0.8, edgecolor=None)
                ax.text(x_buy_broker, y, broker, color=broker_color, fontsize=broker_fontsize, fontweight=font_weight, ha='left', va='center',fontproperties=font_prop)

                x_vol = center_ref - base_offset
                ax.text(x_vol, y, volume_text, color=buy_volume_color, fontsize=value_fontsize, fontweight=font_weight, ha='right', va='center',fontproperties=font_prop)
                if price_text:
                    x_price = x_vol - fixed_gap_value
                    ax.text(x_price, y, price_text, color=buy_price_color, fontsize=value_fontsize, fontweight=font_weight, ha='right', va='center', fontproperties=font_prop)

            except Exception as row_e: print(f"WARN: Error drawing buy row {i+1}: {row_e}")

        # Sell side
        if i < num_sell:
            try:
                broker = format_broker_name(sell_top_raw['券商'].iloc[i])
                volume_val = sell_top_raw['淨賣出'].iloc[i]
                price_val = sell_top_raw['賣出價'].iloc[i]
                price_text, volume_text = format_volume_with_price_label(volume_val, price_val)

                volume_k = abs(volume_val / 1000)
                bar_width = (volume_k / max_abs_volume_k) * bar_max_relative_width if max_abs_volume_k > 0 else 0
                bar_left = center_ref

                ax.barh(y, width=bar_width, left=bar_left, height=bar_height, color=sell_color_bar, alpha=0.8, edgecolor=None)
                ax.text(x_sell_broker, y, broker, color=broker_color, fontsize=broker_fontsize, fontweight=font_weight, ha='right', va='center',fontproperties=font_prop)

                x_vol = center_ref + base_offset
                ax.text(x_vol, y, volume_text, color=sell_volume_color, fontsize=value_fontsize, fontweight=font_weight, ha='left', va='center',fontproperties=font_prop)
                if price_text:
                    x_price = x_vol + fixed_gap_value
                    ax.text(x_price, y, price_text, color=sell_price_color, fontsize=value_fontsize, fontweight=font_weight, ha='left', va='center',fontproperties=font_prop)

            except Exception as row_e: print(f"WARN: Error drawing sell row {i+1}: {row_e}")
    print("DEBUG: Data rows drawn.")

    # --- Draw Summary Text ---
    print("DEBUG: Drawing summary text...")
    total_buy_k = buy_top_raw['淨買入'].sum() / 1000 if num_buy > 0 else 0
    total_sell_k = sell_top_raw['淨賣出'].sum() / 1000 if num_sell > 0 else 0
    summary_text = f'Top{n_items} 總買超: {total_buy_k:,.0f} 張       Top{n_items} 總賣超: {total_sell_k:,.0f} 張'
    summary_y = max_rows + 0.9
    ax.text(0.5, summary_y, summary_text, ha='center', va='center', color=summary_color, fontsize=summary_fontsize, fontweight=font_weight,fontproperties=font_prop)
    print("DEBUG: Summary text drawn.")

    # --- Finalize and Save ---
    print("DEBUG: Adjusting layout and saving figure...")
    plt.tight_layout(pad=0.8)
    try:
        plt.savefig(output_file, facecolor=fig.get_facecolor(), edgecolor='none', dpi=300)
        print(f"SUCCESS: Visualization (v23) saved to: {output_file}")
        return fig
    except Exception as e:
        print(f"ERROR: Failed to save visualization '{output_file}': {e}")
    finally:
         print("DEBUG: Closing figure...")
         plt.close(fig)
         print("DEBUG: Figure closed.")





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
    div[data-baseweb="select"] {
        max-height: 300px;
        overflow-y: auto;
    }
    </style>
""", unsafe_allow_html=True)

# --- 頁面標題 ---
st.title("買賣日報表彙總分析")
st.caption("每日籌碼可至 https://bsr.twse.com.tw/bshtm/ 下載")

today = datetime.today()

# 初始化 report_date（第一次進入）
if "report_date" not in st.session_state:
    st.session_state.report_date = today
    st.session_state.user_selected = False

# 使用者已經選過了，就記得，不再自動更新
selected_date = st.date_input("選擇報表日期", key="report_date")

# 若選擇跟 today 不同，就代表他選過了
if not st.session_state.user_selected and selected_date != today:
    st.session_state.user_selected = True

# ✅ 若沒選過，且日期過了一天，就自動更新為今天
if not st.session_state.user_selected and st.session_state.report_date != today:
    st.session_state.report_date = today

# 日期字串轉換
date = st.session_state.report_date.strftime("%Y-%m-%d")
st.caption(f"📅 目前報表日期為：{date}")

# --- 上傳CSV ---
uploaded_file = st.file_uploader("上傳CSV檔案", type=["csv"])
if uploaded_file is not None:
    df2 = test.csv2df(uploaded_file)
    df_raw = test.df2clean(df2)
    df = test.df2calculate(df_raw)
    
    if df is not None:
        st.success("檔案已整理完成！")

        # 🔵 篩選功能
        st.subheader("原始資料篩選區")

        # 價格範圍（手動輸入）
        min_price_raw = float(df_raw['價格'].min())
        max_price_raw = float(df_raw['價格'].max())

        col1, col2 = st.columns(2)
        with col1:
            price_min_raw = st.number_input(
                "原始資料 - 輸入最小價格",
                min_value=min_price_raw,
                max_value=max_price_raw,
                value=min_price_raw,
                step=0.5,
                format="%.2f",
                key="price_min_raw"
            )
        with col2:
            price_max_raw = st.number_input(
                "原始資料 - 輸入最大價格",
                min_value=min_price_raw,
                max_value=max_price_raw,
                value=max_price_raw,
                step=0.5,
                format="%.2f",
                key="price_max_raw"
            )

        # 券商名稱選擇（可搜尋且加捲軸）
        all_brokers_raw = df_raw['券商'].dropna().unique().tolist()
        selected_brokers_raw = st.multiselect(
            "原始資料 - 選擇券商（可複選）",
            options=all_brokers_raw,
            key="brokers_raw"
        )

        # 🔵 原始資料篩選（用 df_raw）
        df_raw_filtered = df_raw[
            (df_raw['價格'] >= price_min_raw) & 
            (df_raw['價格'] <= price_max_raw)
        ]
        if selected_brokers_raw:
            df_raw_filtered = df_raw_filtered[df_raw_filtered['券商'].isin(selected_brokers_raw)]

        # --- 顯示原始資料 ---
        st.subheader("原始資料")
        st.dataframe(df_raw_filtered, use_container_width=True)

        # CSV 下載按鈕（原本的）
        csv_raw_filtered = df_raw_filtered.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="下載原始資料 CSV",
            data=csv_raw_filtered,
            file_name=f'原始資料_{date}.csv',
            mime='text/csv'
        )
        st.divider()


        # --- 彙整資料篩選區 ---
        st.subheader("彙整資料篩選區")

        min_price_agg = float(df['買入價'].min())
        max_price_agg = float(df['買入價'].max())

        col1, col2 = st.columns(2)
        with col1:
            price_min_agg = st.number_input(
                "彙整資料 - 輸入最小價格",
                min_value=min_price_agg,
                max_value=max_price_agg,
                value=min_price_agg,
                step=0.5,
                format="%.2f",
                key="price_min_agg"
            )
        with col2:
            price_max_agg = st.number_input(
                "彙整資料 - 輸入最大價格",
                min_value=min_price_agg,
                max_value=max_price_agg,
                value=max_price_agg,
                step=0.5,
                format="%.2f",
                key="price_max_agg"
            )

        all_brokers_agg = df['券商'].dropna().unique().tolist()
        selected_brokers_agg = st.multiselect(
            "彙整資料 - 選擇券商（可複選）",
            options=all_brokers_agg,
            key="brokers_agg"
        )

        df_filtered = df[
            (df['買入價'] >= price_min_agg) & 
            (df['買入價'] <= price_max_agg)
        ]
        if selected_brokers_agg:
            df_filtered = df_filtered[df_filtered['券商'].isin(selected_brokers_agg)]

        st.subheader("彙整資料")
        st.dataframe(df_filtered, use_container_width=True)
        
        # CSV 下載按鈕（原本的）
        csv_filtered = df_filtered.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="下載彙整資料 CSV",
            data=csv_filtered,
            file_name=f'彙整資料_{date}.csv',
            mime='text/csv'
        )

        st.divider()



        # --- Top20 報表 + 下載 ---

        ## 📈 買超前20名
        st.subheader("📈 買超前20名")
        df_buy = top20_buy(df)  # 這裡是你原本的邏輯
        st.table(df_buy)

        # CSV 下載
        csv_buy = df_buy.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="下載買超前20名 CSV",
            data=csv_buy,
            file_name=f'買超前20名_{date}.csv',
            mime='text/csv'
        )

        # PNG 下載
        png_buf_buy = df_to_png_bytes(df_buy, "買超前20名", date)
        st.download_button(
            label="下載買超前20名 PNG",
            data=png_buf_buy,
            file_name=f'買超前20名_{date}.png',
            mime='image/png'
        )

        st.divider()


        ## 📉 賣超前20名
        st.subheader("📉 賣超前20名")
        df_sell = top20_sell(df)  # 這裡是你原本的邏輯
        st.table(df_sell)

        # CSV 下載
        csv_sell = df_sell.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="下載賣超前20名 CSV",
            data=csv_sell,
            file_name=f'賣超前20名_{date}.csv',
            mime='text/csv'
        )

        # PNG 下載
        png_buf_sell = df_to_png_bytes(df_sell, "賣超前20名", date)
        st.download_button(
            label="下載賣超前20名 PNG",
            data=png_buf_sell,
            file_name=f'賣超前20名_{date}.png',
            mime='image/png'
        )

        st.divider()


        ## ⚡ 當沖前20名
        st.subheader("⚡ 當沖前20名")
        df_intraday = top20_intraday(df)  # 這裡是你原本的邏輯
        st.table(df_intraday)

        # CSV 下載
        csv_intraday = df_intraday.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="下載當沖前20名 CSV",
            data=csv_intraday,
            file_name=f'當沖前20名_{date}.csv',
            mime='text/csv'
        )

        # PNG 下載
        png_buf_intraday = df_to_png_bytes(df_intraday, "當沖前20名", date)
        st.download_button(
            label="下載當沖前20名 PNG",
            data=png_buf_intraday,
            file_name=f'當沖前20名_{date}.png',
            mime='image/png'
        )

        st.divider()
        

        ## 圖片

        st.subheader("🚀 買賣超對照圖(感謝 B大 大力協助 🙏)")
        st.caption("🎉特別感謝B大🎉 提供此圖表程式碼")
        fig = create_visualization(df_buy, df_sell, date)

        # 將圖形儲存到 BytesIO
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        buf.seek(0)

        # 顯示圖片
        st.image(buf, caption="📷 買賣超對照圖", use_container_width=True)

        # 下載按鈕
        st.download_button(
            label="下載買賣超對照圖 PNG",
            data=buf,
            file_name=f"買賣超對照圖_{date}.png",
            mime="image/png"
        )
