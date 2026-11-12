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


def render_daily_prices(daily):
    high = daily.loc[daily["max"].idxmax()]
    low = daily.loc[daily["min"].idxmin()]
    average = daily["close"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("當月最高價", f"{high['max']:,.2f} 元")
    c2.metric("當月最低價", f"{low['min']:,.2f} 元")
    c3.metric("當月平均股價", f"{average:,.2f} 元")
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
    (show_ma3, "MA3", "#2563eb", 2.6),
    (show_ma12, "MA12", "#dc2626", 3.0),
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

st.subheader("所選月份的股價走勢")
price_months = pd.date_range(first_month, now.replace(day=1), freq="MS")[::-1]
selected_month = st.selectbox(
    "交易月份", price_months,
    index=price_months.get_loc(latest_month),
    format_func=lambda month: f"{month:%Y-%m}",
    key=f"price-month-{stock_id}-{years}",
)
if not daily_prices.empty:
    month_prices = daily_prices.loc[trading_months == selected_month.to_period("M")]
    if month_prices.empty:
        st.info("所選月份沒有可用的成交資料。")
    else:
        render_daily_prices(month_prices)
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
