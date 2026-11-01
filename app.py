"""台股月營收分析：FinMind + Streamlit + Plotly。"""
import os
from datetime import date
from io import StringIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dateutil.relativedelta import relativedelta
from FinMind.data import DataLoader

st.set_page_config(page_title="台股月營收趨勢分析", page_icon="📈", layout="wide")
st.title("📈 台股月營收趨勢分析")
st.caption("FinMind 資料｜月營收、3 個月移動平均（MA3）、12 個月移動平均（MA12）")


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

    df = df[["stock_id", "stock_name"]].copy()
    df = df.dropna().drop_duplicates()
    df["stock_id"] = df["stock_id"].astype(str).str.strip()
    df["stock_name"] = df["stock_name"].astype(str).str.strip()

    return df


def resolve_stock(query, stock_info):
    query = query.strip()

    # 先尋找完全相同的股票代號
    matches = stock_info[
        stock_info["stock_id"] == query
    ]

    # 找不到代號，再以公司名稱搜尋
    if matches.empty:
        matches = stock_info[
            stock_info["stock_name"] == query
        ]

    if matches.empty:
        return None, None

    stock = matches.iloc[0]

    return stock["stock_id"], stock["stock_name"]

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_revenue(stock_id: str, start_date: str, end_date: str, token: str) -> pd.DataFrame:
    api = DataLoader()
    if token:
        api.login_by_token(api_token=token)
    raw = api.taiwan_stock_month_revenue(
        stock_id=stock_id, start_date=start_date, end_date=end_date
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    required = {"revenue_year", "revenue_month", "revenue"}
    if not required.issubset(raw.columns):
        raise ValueError(f"FinMind 回傳資料缺少欄位：{required - set(raw.columns)}")
    raw = raw.copy()
    raw["month"] = pd.to_datetime(
        dict(year=raw["revenue_year"].astype(int),
             month=raw["revenue_month"].astype(int), day=1)
    )
    raw["revenue"] = pd.to_numeric(raw["revenue"], errors="coerce")
    # 同一營收月份若有多筆，以較晚公布的一筆為準。
    if "date" in raw.columns:
        raw = raw.sort_values(["month", "date"])
    else:
        raw = raw.sort_values("month")
    raw = raw.drop_duplicates("month", keep="last").set_index("month")
    raw = raw[["revenue"]].sort_index().asfreq("MS")
    raw["月營收"] = raw["revenue"] / 100_000_000  # 新台幣億元
    raw["MA3"] = raw["月營收"].rolling(3, min_periods=3).mean()
    raw["MA12"] = raw["月營收"].rolling(12, min_periods=12).mean()
    raw["月營收YoY"] = raw["月營收"].pct_change(12, fill_method=None) * 100
    raw["MA3 YoY"] = raw["MA3"].pct_change(12, fill_method=None) * 100
    return raw


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_monthly_price(
    stock_id, start_date, end_date, token
):
    api = DataLoader()

    if token:
        api.login_by_token(api_token=token)

    # 取得每日股價
    daily = api.taiwan_stock_daily(
        stock_id=stock_id,
        start_date=start_date,
        end_date=end_date
    )

    if daily is None or daily.empty:
        return pd.DataFrame()

    daily = daily.copy()

    daily["date"] = pd.to_datetime(daily["date"])

    for col in ["max", "min", "close"]:
        daily[col] = pd.to_numeric(
            daily[col], errors="coerce"
        )

    daily = daily.dropna(
        subset=["max", "min", "close"]
    )

    # 排除無效報價
    daily = daily[
        (daily["max"] > 0) &
        (daily["min"] > 0) &
        (daily["close"] > 0)
    ]

    # 將每日股價彙整為每月股價
    daily["month"] = (
        daily["date"].dt.to_period("M").dt.to_timestamp()
    )

    monthly = daily.groupby("month").agg(
        月最高價=("max", "max"),
        月最低價=("min", "min"),
        月平均股價=("close", "mean")
    )

    return monthly.sort_index()

with st.sidebar:
    st.header("查詢設定")
    stock_query = st.text_input("股票代號或公司名稱", value="2330", placeholder="例如：2330、台積電、聯發科").strip()
    years = st.selectbox("顯示期間", [2, 3, 4, 5, 10], index=2, format_func=lambda x: f"最近 {x} 年")
    show_revenue = st.checkbox("月營收", value=True)
    show_ma3 = st.checkbox("MA3", value=True)
    show_ma12 = st.checkbox("MA12", value=True)
    show_yoy = st.checkbox("另顯示月營收 YoY 圖", value=False)
    refresh = st.button("重新取得資料", use_container_width=True)
    st.caption("如有 FinMind Token，可設定環境變數 FINMIND_TOKEN 或 Streamlit secrets。")

if refresh:
    fetch_revenue.clear()
    fetch_monthly_price.clear()


if not stock_query:
    st.warning("請輸入股票代號或公司名稱")
    st.stop()

try:
    stock_info = fetch_stock_info(get_token())

    stock_id, stock_name = resolve_stock(
        stock_query, stock_info
    )

except Exception as exc:
    st.error(f"取得股票名稱清單失敗：{exc}")
    st.stop()

if stock_id is None:
    st.warning(
        f"找不到「{stock_query}」，"
        "請確認股票代號或公司名稱。"
    )
    st.stop()

st.sidebar.success(
    f"已辨識：{stock_id} {stock_name}"
)


now = date.today()
first_month = now.replace(day=1) - relativedelta(months=years * 12 - 1)
# 為了在顯示區間第一個月有完整 MA12，並計算 MA3 YoY，往前抓 23 個月。
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

# 排除未來營收月份；未公布月份不補零。
df = df.loc[df.index <= pd.Timestamp(now.replace(day=1))]
view = df.loc[df.index >= pd.Timestamp(first_month)].copy()
valid = view.dropna(subset=["月營收"])
if valid.empty:
    st.warning("選定期間內沒有可用營收資料。")
    st.stop()
latest_month = valid.index[-1]
latest = valid.iloc[-1]

st.subheader(
    f"{stock_id} {stock_name}｜"
    f"截至 {latest_month:%Y-%m} 的已取得營收"
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("最新月營收（億元）", f"{latest['月營收']:,.2f}")
c2.metric("MA3（億元）", f"{latest['MA3']:,.2f}" if pd.notna(latest["MA3"]) else "—")
c3.metric("MA12（億元）", f"{latest['MA12']:,.2f}" if pd.notna(latest["MA12"]) else "—")
c4.metric("月營收 YoY", f"{latest['月營收YoY']:+.2f}%" if pd.notna(latest["月營收YoY"]) else "—")

st.markdown(
    f"### {stock_id} {stock_name}｜月營收與移動平均"
)

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
    title=None, xaxis_title="營收月份",
    yaxis_title="新台幣億元", hovermode="x unified", height=550,
    legend=dict(orientation="h", y=1.08), margin=dict(l=40, r=30, t=35, b=50),
)
fig.update_xaxes(tickformat="%Y-%m", rangeslider_visible=True)
st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": "hover",
        "displaylogo": False
    }
)
st.caption("可使用圖例開關曲線、滑鼠查看數值，或拖曳下方時間軸縮放。")


# ===== 月股價趨勢圖 =====

st.markdown(
    f"### {stock_id} {stock_name}｜月股價趨勢"
)

try:
    with st.spinner("正在取得每日股價資料..."):
        price_df = fetch_monthly_price(
            stock_id,
            first_month.isoformat(),
            now.isoformat(),
            get_token()
        )

except Exception as exc:
    st.warning(f"股價資料取得失敗：{exc}")
    price_df = pd.DataFrame()

if not price_df.empty:

    price_view = price_df.loc[
        (price_df.index >= pd.Timestamp(first_month)) &
        (price_df.index <= pd.Timestamp(now))
    ].copy()

    price_fig = go.Figure()

    price_lines = [
        ("月最高價", "#dc2626"),
        ("月平均股價", "#2563eb"),
        ("月最低價", "#16a34a")
    ]

    for column, color in price_lines:
        price_fig.add_trace(
            go.Scatter(
                x=price_view.index,
                y=price_view[column],
                mode="lines+markers",
                name=column,
                line=dict(color=color, width=2.5),
                marker=dict(size=4),
                hovertemplate=(
                    "%{x|%Y-%m}<br>"
                    "%{y:,.2f} 元"
                    "<extra>" + column + "</extra>"
                ),
                connectgaps=False
            )
        )

    price_fig.update_layout(
        title=None,
        xaxis_title="交易月份",
        yaxis_title="股價（新台幣元）",
        hovermode="x unified",
        height=500,
        legend=dict(
            orientation="h",
            y=1.08
        ),
        margin=dict(
            l=40, r=30, t=35, b=50
        )
    )

    price_fig.update_xaxes(
        tickformat="%Y-%m",
        rangeslider_visible=True,
        range=[
            pd.Timestamp(first_month),
            pd.Timestamp(now)
        ]
    )

    st.plotly_chart(
        price_fig,
        use_container_width=True,
        config={
            "displayModeBar": "hover",
            "displaylogo": False
        }
    )

    st.caption(
        "月最高價＝當月最高成交價；"
        "月最低價＝當月最低成交價；"
        "月平均股價＝當月每日收盤價算術平均。"
        "採未還原股價，除權息、股票分割等"
        "可能影響跨期比較。"
    )

    # 顯示最近 12 個月股價統計
    with st.expander("查看最近 12 個月股價明細"):
        st.dataframe(
            price_view.tail(12)
            .sort_index(ascending=False)
            .style.format("{:,.2f}"),
            use_container_width=True
        )

else:
    st.info("目前沒有取得可用的股價資料。")

if show_yoy:
    yoy_fig = go.Figure(go.Scatter(
        x=view.index, y=view["月營收YoY"], mode="lines+markers", name="月營收 YoY",
        line=dict(color="#16a34a", width=2), connectgaps=False,
        hovertemplate="%{x|%Y-%m}<br>%{y:+.2f}%<extra></extra>",
    ))
    yoy_fig.add_hline(y=0, line_dash="dash", line_color="gray")
    yoy_fig.update_layout(title="月營收年增率", yaxis_title="%", height=350)
    st.plotly_chart(yoy_fig, use_container_width=True)

st.subheader("月營收明細")
output = view[["月營收", "MA3", "MA12", "月營收YoY", "MA3 YoY"]].copy()
output.index.name = "營收月份"
st.dataframe(output.sort_index(ascending=False).style.format("{:,.2f}", na_rep="—"), use_container_width=True)

csv = output.to_csv(encoding="utf-8-sig", float_format="%.4f")
a, b = st.columns(2)
with a:
    st.download_button("下載 CSV", csv.encode("utf-8-sig"),
                       file_name=f"{stock_id}_monthly_revenue.csv", mime="text/csv", use_container_width=True)
with b:
    st.download_button("下載互動式 HTML 圖表", fig.to_html(include_plotlyjs=True).encode("utf-8"),
                       file_name=f"{stock_id}_revenue_chart.html", mime="text/html", use_container_width=True)

st.info("說明：MA3/MA12 為含當月的移動平均；資料以實際營收所屬月份排列。當月份資料尚未公布時，不會用 0 補值。若 FinMind 缺少月份，對應移動平均可能顯示空值。")
