"""台股月營收分析：FinMind + Streamlit + Plotly。

2026-11: 將原側邊欄查詢及顯示設定移至主畫面頂部。
"""
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dateutil.relativedelta import relativedelta
from FinMind.data import DataLoader

st.set_page_config(page_title="台股月營收趨勢分析", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
st.title("📈 台股月營收趨勢分析")
st.caption("FinMind 資料｜月營收、移動平均與股價走勢")
CHART_CONFIG = {"displayModeBar": False, "displaylogo": False}


def get_token():
    try:
        return st.secrets.get("FINMIND_TOKEN", os.getenv("FINMIND_TOKEN", ""))
    except (FileNotFoundError, KeyError):
        return os.getenv("FINMIND_TOKEN", "")


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_stock_info(token):
    api = DataLoader()
    if token:
        api.login_by_token(api_token=token)
    df = api.taiwan_stock_info()
    if df is None or df.empty:
        raise ValueError("無法取得股票名稱清單")
    df = df[["stock_id", "stock_name"]].copy().dropna().drop_duplicates()
    df["stock_id"] = df["stock_id"].astype(str).str.strip()
    df["stock_name"] = df["stock_name"].astype(str).str.strip()
    return df


def resolve_stock(query, stock_info):
    query = query.strip()
    matches = stock_info[stock_info["stock_id"] == query]
    if matches.empty:
        matches = stock_info[stock_info["stock_name"] == query]
    if matches.empty:
        return None, None
    stock = matches.iloc[0]
    return stock["stock_id"], stock["stock_name"]


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_revenue(stock_id: str, start_date: str, end_date: str, token: str) -> pd.DataFrame:
    api = DataLoader()
    if token:
        api.login_by_token(api_token=token)
    raw = api.taiwan_stock_month_revenue(stock_id=stock_id, start_date=start_date, end_date=end_date)
    if raw is None or raw.empty:
        return pd.DataFrame()
    required = {"revenue_year", "revenue_month", "revenue"}
    if not required.issubset(raw.columns):
        raise ValueError(f"FinMind 回傳資料缺少欄位：{required - set(raw.columns)}")
    raw = raw.copy()
    raw["month"] = pd.to_datetime(dict(year=raw["revenue_year"].astype(int), month=raw["revenue_month"].astype(int), day=1))
    raw["revenue"] = pd.to_numeric(raw["revenue"], errors="coerce")
    raw = raw.sort_values(["month", "date"] if "date" in raw.columns else "month")
    raw = raw.drop_duplicates("month", keep="last").set_index("month")
    raw = raw[["revenue"]].sort_index().asfreq("MS")
    raw["月營收"] = raw["revenue"] / 100_000_000
    raw["MA3"] = raw["月營收"].rolling(3, min_periods=3).mean()
    raw["MA12"] = raw["月營收"].rolling(12, min_periods=12).mean()
    raw["月營收YoY"] = raw["月營收"].pct_change(12, fill_method=None) * 100
    raw["MA3 YoY"] = raw["MA3"].pct_change(12, fill_method=None) * 100
    return raw


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_daily_prices(stock_id, start_date, end_date, token):
    api = DataLoader()
    if token:
        api.login_by_token(api_token=token)
    daily = api.taiwan_stock_daily(stock_id=stock_id, start_date=start_date, end_date=end_date)
    if daily is None or daily.empty:
        return pd.DataFrame()
    price_columns = ["max", "min", "close"]
    required = {"date", *price_columns}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"FinMind 股價資料缺少欄位：{', '.join(sorted(missing))}")
    daily = daily[["date", *price_columns]].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    for column in price_columns:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    in_range = daily["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    valid_prices = daily[price_columns].gt(0).all(axis=1)
    return (daily.loc[in_range & valid_prices]
            .sort_values("date", kind="stable")
            .drop_duplicates("date", keep="last"))



@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_foreign_investors(stock_id, start_date, end_date, token):
    """取得外資（不含外資自營商）每日買賣超，原始單位為股。"""
    api = DataLoader()
    if token:
        api.login_by_token(api_token=token)
    raw = api.taiwan_stock_institutional_investors(
        stock_id=stock_id, start_date=start_date, end_date=end_date
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    required = {"date", "name", "buy", "sell"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"FinMind 法人資料缺少欄位：{', '.join(sorted(missing))}")
    foreign = raw.loc[raw["name"] == "Foreign_Investor", ["date", "buy", "sell"]].copy()
    if foreign.empty:
        return pd.DataFrame()
    foreign["date"] = pd.to_datetime(foreign["date"], errors="coerce").dt.normalize()
    foreign["buy"] = pd.to_numeric(foreign["buy"], errors="coerce")
    foreign["sell"] = pd.to_numeric(foreign["sell"], errors="coerce")
    foreign = foreign.dropna(subset=["date", "buy", "sell"])
    foreign = foreign.groupby("date", as_index=False)[["buy", "sell"]].sum()
    foreign["當日外資買賣超張數"] = (foreign["buy"] - foreign["sell"]) / 1000
    return foreign[["date", "當日外資買賣超張數"]].sort_values("date")


def render_foreign_cumulative(stock_id, stock_name, start_date, end_date, period_prices, token, trading_days):
    """所選起訖日期內，每個交易日截至當日往前指定交易日數的外資淨買賣超。"""
    st.subheader(f"最近{trading_days}個交易日外資累計買賣超張數")
    if period_prices.empty:
        st.info("所選日期範圍沒有股價交易日，無法對照外資買賣超。")
        return

    # 依使用者設定的交易日數向前抓取足夠的日曆天數（包含假日緩衝）。
    lookback_calendar_days = max(120, int(trading_days * 2.2) + 45)
    start = start_date - relativedelta(days=lookback_calendar_days)
    end = end_date
    try:
        with st.spinner(f"正在計算所選日期範圍每日的{trading_days}交易日外資買賣超..."):
            foreign = fetch_foreign_investors(
                stock_id, start.isoformat(), end.isoformat(), token
            )
            history_prices = fetch_daily_prices(
                stock_id, start.isoformat(), end.isoformat(), token
            )
    except Exception as exc:
        st.warning(f"外資買賣超資料取得失敗：{exc}")
        return

    if foreign.empty or history_prices.empty:
        st.info("目前沒有可用的外資買賣超或歷史股價資料。")
        return

    # 以實際股價交易日作為滾動視窗基準，避免漏報的法人資料造成日期錯位。
    history = history_prices[["date", "close"]].merge(
        foreign, on="date", how="left", validate="one_to_one"
    ).sort_values("date").reset_index(drop=True)
    rolling_col = f"{trading_days}日外資買賣超張數"
    valid_col = f"{trading_days}日有效資料筆數"
    history[rolling_col] = history["當日外資買賣超張數"].rolling(
        window=trading_days, min_periods=trading_days
    ).sum()
    history[valid_col] = history["當日外資買賣超張數"].rolling(
        window=trading_days, min_periods=trading_days
    ).count()
    # 缺少任何一日法人資料時不顯示不完整的30日合計。
    history.loc[history[valid_col] < trading_days, rolling_col] = float("nan")
    mask = history["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    shown = history.loc[mask].copy()
    if shown.empty:
        st.info("所選日期範圍沒有可對照的交易日。")
        return

    # 依所選日期範圍「實際顯示」的30日外資買賣超有效值設定左側Y軸。
    foreign_visible = shown[rolling_col].dropna()
    foreign_axis_range = None
    if not foreign_visible.empty:
        low = float(foreign_visible.min())
        high = float(foreign_visible.max())
        if high == low:
            # 全部相同時仍保留可辨識的上下空間，避免零高度座標軸。
            padding = max(abs(high) * 0.05, 1.0)
        else:
            padding = (high - low) * 0.05
        foreign_axis_range = [low - padding, high + padding]

    fig_foreign = go.Figure()
    fig_foreign.add_trace(go.Scatter(
        x=shown["date"], y=shown[rolling_col],
        name=f"近{trading_days}交易日外資買賣超（張）", mode="lines+markers",
        line=dict(color="#dc2626", width=2.8), marker=dict(size=5),
        hovertemplate=f"%{{x|%Y-%m-%d}}<br>{trading_days}日合計 %{{y:+,.1f}} 張<extra>外資</extra>",
        connectgaps=False,
    ))
    fig_foreign.add_trace(go.Scatter(
        x=shown["date"], y=shown["close"],
        name="每日收盤價（元）", mode="lines+markers", yaxis="y2",
        line=dict(color="#2563eb", width=2.2), marker=dict(size=4),
        hovertemplate="%{x|%Y-%m-%d}<br>收盤 %{y:,.2f} 元<extra>股價</extra>",
    ))
    fig_foreign.add_hline(y=0, line_dash="dot", line_color="#9ca3af")
    fig_foreign.update_layout(
        xaxis_title="交易日期", yaxis=dict(title=f"近{trading_days}交易日外資買賣超（張）", range=foreign_axis_range),
        yaxis2=dict(title="收盤價（元）", overlaying="y", side="right", showgrid=False),
        hovermode="x unified", height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=55, r=55, t=45, b=50),
    )
    fig_foreign.update_xaxes(type="date", tickformat="%m/%d")
    st.plotly_chart(fig_foreign, use_container_width=True, config=CHART_CONFIG)
    st.caption(
        f"{stock_id} {stock_name}｜{start_date:%Y-%m-%d} 至 {end_date:%Y-%m-%d} 各交易日，"
        f"分別計算截至當日（含當日）往前{trading_days}個交易日的外資買賣超總和。"
        "外資買賣超＝（買進股數－賣出股數）÷1,000，未包含外資自營商。"
        "左右軸單位不同，請比較趨勢而非線條高度。"
    )
    missing = shown[rolling_col].isna().sum()
    if missing:
        st.warning(f"所選日期範圍有 {missing} 個交易日缺少完整{trading_days}日法人資料，對應數值留空，避免誤導。")
    with st.expander(f"查看所選日期範圍每日{trading_days}交易日外資買賣超明細"):
        detail = shown[["date", "close", "當日外資買賣超張數", rolling_col]].rename(
            columns={"date": "交易日期", "close": "收盤價（元）"}
        )
        st.dataframe(
            detail.iloc[::-1].style.format({
                "收盤價（元）": "{:,.2f}",
                "當日外資買賣超張數": "{:+,.1f}",
                rolling_col: "{:+,.1f}",
            }, na_rep="—"), use_container_width=True, hide_index=True
        )



def render_predictive_test(stock_id, stock_name, start_date, end_date, token, trading_days):
    """在所選日期範圍內，檢驗近N日外資買賣超與未來N交易日報酬的關係。"""
    st.subheader(f"外資近{trading_days}日買賣超 vs. 未來{trading_days}交易日報酬率")
    st.caption("訊號日 t：截至 t（含當日）的近 N 個交易日外資淨買賣超；"
               "結果：從 t 日收盤到第 N 個後續交易日收盤的報酬率。"
               "僅採用未來價格已知的歷史樣本。")

    # 前方須有 N 個交易日，後方須有 N 個交易日；後方查到今天即可。
    calendar_buffer = max(120, int(trading_days * 2.2) + 45)
    fetch_start = start_date - relativedelta(days=calendar_buffer)
    today = datetime.now(ZoneInfo("Asia/Taipei")).date()
    fetch_end = min(today, end_date + relativedelta(days=calendar_buffer))
    try:
        with st.spinner("正在計算外資訊號與未來報酬..."):
            prices = fetch_daily_prices(stock_id, fetch_start.isoformat(),
                                        fetch_end.isoformat(), token)
            foreign = fetch_foreign_investors(stock_id, fetch_start.isoformat(),
                                              min(end_date, today).isoformat(), token)
    except Exception as exc:
        st.warning(f"回測資料取得失敗：{exc}")
        return
    if prices.empty or foreign.empty:
        st.info("沒有足夠的股價或外資資料可供回測。")
        return

    df = prices[["date", "close"]].merge(
        foreign, on="date", how="left", validate="one_to_one"
    ).sort_values("date").reset_index(drop=True)
    col = f"近{trading_days}日外資淨買賣超（張）"
    ret_col = f"未來{trading_days}日報酬率（%）"
    df[col] = df["當日外資買賣超張數"].rolling(
        trading_days, min_periods=trading_days
    ).sum()
    df[ret_col] = (df["close"].shift(-trading_days) / df["close"] - 1) * 100
    df["未來到期日"] = df["date"].shift(-trading_days)
    df = df.loc[df["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))]
    df = df.dropna(subset=[col, ret_col, "未來到期日"]).copy()

    if len(df) < 5:
        st.info("有效樣本少於5筆。請擴大日期區間，或減少 N；"
                "接近今天的日期因未來 N 日尚未發生，會自動排除。")
        return

    pearson = df[col].corr(df[ret_col], method="pearson")
    spearman = df[col].corr(df[ret_col], method="spearman")
    positive = df[col] > 0
    negative = df[col] < 0
    avg_pos = df.loc[positive, ret_col].mean()
    avg_neg = df.loc[negative, ret_col].mean()
    a, b, c, d = st.columns(4)
    a.metric("有效訊號日", f"{len(df)}")
    b.metric("Pearson 相關係數", f"{pearson:.3f}" if pd.notna(pearson) else "無法計算")
    c.metric("Spearman 等級相關", f"{spearman:.3f}" if pd.notna(spearman) else "無法計算")
    d.metric("買超日後續平均報酬", f"{avg_pos:+.2f}%" if pd.notna(avg_pos) else "無樣本")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[col], y=df[ret_col], mode="markers", name="歷史訊號日",
        marker=dict(color="#dc2626", size=8, opacity=0.65),
        customdata=df["date"].dt.strftime("%Y-%m-%d"),
        hovertemplate="訊號日 %{customdata}<br>外資 %{x:+,.0f} 張"
                      "<br>未來報酬 %{y:+.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#9ca3af")
    fig.add_vline(x=0, line_dash="dot", line_color="#9ca3af")
    fig.update_layout(
        xaxis_title=col, yaxis_title=ret_col, height=430,
        margin=dict(l=50, r=25, t=25, b=55),
    )
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    summary = pd.DataFrame({
        "訊號類別": ["近N日累計買超（>0）", "近N日累計賣超（<0）"],
        "樣本數": [int(positive.sum()), int(negative.sum())],
        "未來N日平均報酬（%）": [avg_pos, avg_neg],
        "未來N日上漲比例（%）": [
            df.loc[positive, ret_col].gt(0).mean() * 100 if positive.any() else float("nan"),
            df.loc[negative, ret_col].gt(0).mean() * 100 if negative.any() else float("nan"),
        ],
    })
    st.dataframe(summary.style.format({
        "未來N日平均報酬（%）": "{:+.2f}",
        "未來N日上漲比例（%）": "{:.1f}",
    }, na_rep="—"), use_container_width=True, hide_index=True)

    if pd.notna(avg_pos) and pd.notna(avg_neg):
        st.write(f"買超訊號與賣超訊號的未來平均報酬差：**{avg_pos - avg_neg:+.2f} 個百分點**。")

    with st.expander("查看逐日回測資料與下載 CSV"):
        detail = df[["date", col, "close", "未來到期日", ret_col]].rename(
            columns={"date": "訊號日", "close": "訊號日收盤價"}
        )
        st.dataframe(detail.iloc[::-1], use_container_width=True, hide_index=True)
        st.download_button(
            "下載回測資料 CSV",
            detail.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{stock_id}_foreign_{trading_days}d_forward_return.csv",
            mime="text/csv",
        )

    st.warning(
        "這是探索性相關分析，不等於已證實有預測力。相鄰日期的 N 日視窗高度重疊，"
        "樣本不獨立，相關係數及平均報酬可能被趨勢與極端值影響；"
        "此外使用當日外資資料與當日收盤價計算，並不代表可在當日收盤前取得完整訊號。"
        "報酬採未還原收盤價，不含股利、交易成本與滑價。"
        "建議進一步用不重疊樣本、不同市場階段與樣本外期間驗證。"
    )


def render_daily_prices(daily):
    high = daily.loc[daily["max"].idxmax()]
    low = daily.loc[daily["min"].idxmin()]
    average = daily["close"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("區間最高價", f"{high['max']:,.2f} 元")
    c2.metric("區間最低價", f"{low['min']:,.2f} 元")
    c3.metric("區間平均股價", f"{average:,.2f} 元")
    chart = go.Figure(go.Scatter(
        x=daily["date"], y=daily["close"], name="每日收盤價", mode="lines+markers",
        line=dict(color="#2563eb", width=2.3), marker=dict(size=5),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.2f} 元<extra>每日收盤價</extra>",
    ))
    for label, point, field, color, symbol in [
        ("當月最高價", high, "max", "#b8384b", "triangle-up"),
        ("當月最低價", low, "min", "#008068", "triangle-down"),
    ]:
        chart.add_trace(go.Scatter(
            x=[point["date"]], y=[point[field]], name=label, mode="markers",
            marker=dict(color=color, symbol=symbol, size=12),
            hovertemplate=f"%{{x|%Y-%m-%d}}<br>%{{y:,.2f}} 元<extra>{label}</extra>",
        ))
    chart.add_hline(y=average, line_color="#895c12", line_dash="dot", name="平均收盤價", showlegend=True)
    chart.update_layout(
        xaxis_title="交易日期", yaxis_title="股價（新台幣元）", height=370,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=30, t=45, b=50),
    )
    chart.update_xaxes(type="date", tickformat="%m/%d")
    st.plotly_chart(chart, use_container_width=True, config=CHART_CONFIG)
    st.caption(f"共 {len(daily)} 個有報價的交易日。高低價採盤中成交價，平均股價是每日收盤價的算術平均。資料為未還原股價。")


# ===== 原 Sidebar 全部移至主畫面頂部 =====
st.subheader("查詢設定")
col_stock, col_years, col_refresh = st.columns([3, 2, 1], vertical_alignment="bottom")
with col_stock:
    stock_query = st.text_input("股票代號或公司名稱", value="2330", placeholder="例如：2330、台積電、聯發科").strip()
with col_years:
    years = st.selectbox("顯示期間", [2, 3, 4, 5, 10], index=2, format_func=lambda x: f"最近 {x} 年")
with col_refresh:
    refresh = st.button("重新取得資料", use_container_width=True)

show_col1, show_col2, show_col3, show_col4 = st.columns(4)
with show_col1:
    show_revenue = st.checkbox("月營收", value=True)
with show_col2:
    show_ma3 = st.checkbox("MA3", value=True)
with show_col3:
    show_ma12 = st.checkbox("MA12", value=True)
with show_col4:
    show_yoy = st.checkbox("另顯示月營收 YoY 圖", value=False)
st.caption("如有 FinMind Token，可在 Streamlit Cloud 的 Secrets 設定 FINMIND_TOKEN。")

if refresh:
    fetch_revenue.clear()
    fetch_daily_prices.clear()
    fetch_foreign_investors.clear()
    fetch_stock_info.clear()

if not stock_query:
    st.warning("請輸入股票代號或公司名稱")
    st.stop()
try:
    stock_info = fetch_stock_info(get_token())
    stock_id, stock_name = resolve_stock(stock_query, stock_info)
except Exception as exc:
    st.error(f"取得股票名稱清單失敗：{exc}")
    st.stop()
if stock_id is None:
    st.warning(f"找不到「{stock_query}」，請確認股票代號或公司名稱。")
    st.stop()
st.success(f"已辨識：{stock_id} {stock_name}")
st.divider()

now = datetime.now(ZoneInfo("Asia/Taipei")).date()
first_month = now.replace(day=1) - relativedelta(months=years * 12 - 1)
fetch_start = first_month - relativedelta(months=23)
try:
    with st.spinner("正在從 FinMind 取得月營收資料..."):
        df = fetch_revenue(stock_id, fetch_start.isoformat(), now.isoformat(), get_token())
except Exception as exc:
    st.error(f"資料取得失敗：{exc}")
    st.info("請確認網路、FinMind API 額度、股票代號及 Token 設定。")
    st.stop()
if df.empty or df["月營收"].dropna().empty:
    st.warning("沒有取得月營收資料。請檢查股票代號或 FinMind 資料可用性。")
    st.stop()
df = df.loc[df.index <= pd.Timestamp(now.replace(day=1))]
view = df.loc[df.index >= pd.Timestamp(first_month)].copy()
valid = view.dropna(subset=["月營收"])
if valid.empty:
    st.warning("選定期間內沒有可用營收資料。")
    st.stop()
latest_month = valid.index[-1]
latest = valid.iloc[-1]
st.subheader(f"{stock_id} {stock_name}")
st.caption(f"最新已公布營收月份 {latest_month:%Y-%m}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("最新月營收（億元）", f"{latest['月營收']:,.2f}")
c2.metric("MA3（億元）", f"{latest['MA3']:,.2f}" if pd.notna(latest["MA3"]) else "—")
c3.metric("MA12（億元）", f"{latest['MA12']:,.2f}" if pd.notna(latest["MA12"]) else "—")
c4.metric("月營收 YoY", f"{latest['月營收YoY']:+.2f}%" if pd.notna(latest["月營收YoY"]) else "—")

st.subheader("月營收與移動平均")
fig = go.Figure()
for enabled, column, color, width in [
    (show_revenue, "月營收", "#9ca3af", 1.8),
    (show_ma3, "MA3", "#dc2626", 2.6),
    (show_ma12, "MA12", "#2563eb", 3.0),
]:
    if enabled:
        fig.add_trace(go.Scatter(
            x=view.index, y=view[column], mode="lines+markers", name=column,
            line=dict(color=color, width=width), marker=dict(size=3),
            hovertemplate="%{x|%Y-%m}<br>%{y:,.2f} 億元<extra>" + column + "</extra>",
            connectgaps=False,
        ))
fig.update_layout(
    title=None, xaxis_title="營收月份", yaxis_title="新台幣億元",
    hovermode="x unified", height=550,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=40, r=30, t=45, b=50),
)
fig.update_xaxes(tickformat="%Y-%m", rangeslider_visible=True)
st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
st.caption("可使用圖例開關曲線、滑鼠查看數值，或拖曳下方時間軸縮放。")

st.subheader("多月股價趨勢")
try:
    with st.spinner("正在取得每日股價資料..."):
        daily_prices = fetch_daily_prices(stock_id, first_month.isoformat(), now.isoformat(), get_token())
except Exception as exc:
    st.warning(f"股價資料取得失敗：{exc}")
    daily_prices = pd.DataFrame()
if not daily_prices.empty:
    trading_months = daily_prices["date"].dt.to_period("M")
    price_view = daily_prices.groupby(trading_months.dt.to_timestamp()).agg(
        月最高價=("max", "max"), 月最低價=("min", "min"), 月平均股價=("close", "mean"),
    )
    price_view.index.name = "交易月份"
    price_fig = go.Figure()
    for column, color in [
        ("月最高價", "#dc2626"), ("月平均股價", "#2563eb"), ("月最低價", "#16a34a")
    ]:
        price_fig.add_trace(go.Scatter(
            x=price_view.index, y=price_view[column], mode="lines+markers", name=column,
            line=dict(color=color, width=2.5), marker=dict(size=4),
            hovertemplate="%{x|%Y-%m}<br>%{y:,.2f} 元<extra>" + column + "</extra>",
            connectgaps=False,
        ))
    price_fig.update_layout(
        title=None, xaxis_title="交易月份", yaxis_title="股價（新台幣元）",
        hovermode="x unified", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=30, t=45, b=50),
    )
    price_fig.update_xaxes(
        tickformat="%Y-%m", rangeslider_visible=True,
        range=[pd.Timestamp(first_month), pd.Timestamp(now)],
    )
    st.plotly_chart(price_fig, use_container_width=True, config=CHART_CONFIG)
    st.caption("月最高價＝當月最高成交價；月最低價＝當月最低成交價；月平均股價＝當月每日收盤價算術平均。採未還原股價，除權息、股票分割等可能影響跨期比較。")
    with st.expander("查看最近 12 個月股價明細"):
        st.dataframe(price_view.tail(12).sort_index(ascending=False).style.format("{:,.2f}"), use_container_width=True)
else:
    st.info("目前沒有取得可用的股價資料。")

st.subheader("外資買賣超與股價比較｜選擇日期範圍")
# 預設選取最新已公布營收月份；日期範圍可自由跨月。
default_start = max(first_month, latest_month.date())
default_end = min(now, (latest_month + relativedelta(months=1, days=-1)).date())
if default_end < default_start:
    default_start = max(first_month, now - relativedelta(days=30))
    default_end = now
start_col, end_col, days_col = st.columns([2, 2, 1])
with start_col:
    selected_start = st.date_input(
        "起始日期", value=default_start, min_value=first_month,
        max_value=now, key=f"price-start-{stock_id}-{years}",
    )
with end_col:
    selected_end = st.date_input(
        "結束日期", value=default_end, min_value=first_month,
        max_value=now, key=f"price-end-{stock_id}-{years}",
    )
with days_col:
    trading_days = st.number_input(
        "累計交易日數", min_value=1, max_value=250, value=30, step=1,
        help="每個顯示日期分別累計截至當日往前 N 個交易日的外資買賣超。",
    )
if selected_start > selected_end:
    st.warning("起始日期不可晚於結束日期，請重新選擇日期。")
else:
    period_prices = pd.DataFrame()
    if not daily_prices.empty:
        period_prices = daily_prices.loc[
            daily_prices["date"].between(pd.Timestamp(selected_start), pd.Timestamp(selected_end))
        ].copy()
    if period_prices.empty:
        st.info("所選日期範圍沒有可用的成交資料。")
    # 外資圖內已有收盤價折線，不再另外繪製每日股價圖。
    render_foreign_cumulative(
        stock_id, stock_name, selected_start, selected_end, period_prices, get_token(), int(trading_days)
    )
    render_predictive_test(
        stock_id, stock_name, selected_start, selected_end, get_token(), int(trading_days)
    )

if show_yoy:
    yoy_fig = go.Figure(go.Scatter(
        x=view.index, y=view["月營收YoY"], mode="lines+markers", name="月營收 YoY",
        line=dict(color="#16a34a", width=2), connectgaps=False,
        hovertemplate="%{x|%Y-%m}<br>%{y:+.2f}%<extra></extra>",
    ))
    yoy_fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.subheader("月營收年增率")
    yoy_fig.update_layout(yaxis_title="%", height=350)
    st.plotly_chart(yoy_fig, use_container_width=True, config=CHART_CONFIG)
st.subheader("月營收明細")
output = view[["月營收", "MA3", "MA12", "月營收YoY", "MA3 YoY"]].copy()
output.index.name = "營收月份"
st.dataframe(output.sort_index(ascending=False).style.format("{:,.2f}", na_rep="—"), use_container_width=True)
csv = output.to_csv(encoding="utf-8-sig", float_format="%.4f")
a, b = st.columns(2)
with a:
    st.download_button("下載 CSV", csv.encode("utf-8-sig"), file_name=f"{stock_id}_monthly_revenue.csv", mime="text/csv", use_container_width=True)
with b:
    st.download_button("下載互動式 HTML 圖表", fig.to_html(include_plotlyjs=True).encode("utf-8"), file_name=f"{stock_id}_revenue_chart.html", mime="text/html", use_container_width=True)
st.info("說明：MA3/MA12 為含當月的移動平均；資料以實際營收所屬月份排列。當月份資料尚未公布時，不會用 0 補值。若 FinMind 缺少月份，對應移動平均可能顯示空值。")
