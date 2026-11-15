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

    # 將交易日期轉成類別軸標籤：休市日不占用 X 軸間距。
    # 兩條折線共用同一組標籤，確保外資與股價逐日對齊。
    trading_date_labels = shown["date"].dt.strftime("%Y-%m-%d")
    fig_foreign = go.Figure()
    fig_foreign.add_trace(go.Scatter(
        x=trading_date_labels, y=shown[rolling_col],
        name=f"近{trading_days}交易日外資買賣超（張）", mode="lines+markers",
        line=dict(color="#dc2626", width=2.8), marker=dict(size=5),
        hovertemplate=f"%{{x}}<br>{trading_days}日合計 %{{y:+,.1f}} 張<extra>外資</extra>",
        connectgaps=False,
    ))
    fig_foreign.add_trace(go.Scatter(
        x=trading_date_labels, y=shown["close"],
        name="每日收盤價（元）", mode="lines+markers", yaxis="y2",
        line=dict(color="#2563eb", width=2.2), marker=dict(size=4),
        hovertemplate="%{x}<br>收盤 %{y:,.2f} 元<extra>股價</extra>",
    ))
    fig_foreign.add_hline(y=0, line_dash="dot", line_color="#9ca3af")
    fig_foreign.update_layout(
        xaxis_title="交易日期", yaxis=dict(title=f"近{trading_days}交易日外資買賣超（張）", range=foreign_axis_range),
        yaxis2=dict(title="收盤價（元）", overlaying="y", side="right", showgrid=False),
        hovermode="x unified", height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=55, r=55, t=45, b=50),
    )
    fig_foreign.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=trading_date_labels.tolist(),
        tickangle=-45,
        nticks=16,
    )
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



def render_revenue_insights(revenue):
    """以已公布月營收、MA3、MA12，產生可追溯的規則式趨勢歸納。"""
    st.markdown("#### 營收趨勢歸納")
    valid = revenue.dropna(subset=["月營收"]).sort_index()
    if valid.empty:
        st.info("目前沒有足夠的營收資料可供分析。")
        return

    current = valid.iloc[-1]
    month = valid.index[-1]
    notes = []

    def number(value):
        return f"{value:,.2f}"

    def pct(value):
        return f"{value:+.1f}%"

    def change(new, old):
        if pd.isna(new) or pd.isna(old) or old == 0:
            return None
        return (new / old - 1) * 100

    # 月營收與年增率：月增率只在相鄰兩個日曆月份都有資料時才計算。
    notes.append(f"**最新月份（{month:%Y/%m}）**：營收 {number(current['月營收'])} 億元。")
    prev_month = month - pd.DateOffset(months=1)
    if prev_month in revenue.index and pd.notna(revenue.loc[prev_month, "月營收"]):
        mom = change(current["月營收"], revenue.loc[prev_month, "月營收"])
        if mom is not None:
            notes.append(f"**月增率（MoM）**：{pct(mom)}；單月變化可能受到工作天數、出貨節奏及季節性影響。")

    yoy = current["月營收YoY"]
    if pd.notna(yoy):
        notes.append(f"**年增率（YoY）**：{pct(yoy)}，"
                     + ("高於去年同期。" if yoy > 0 else "低於去年同期。" if yoy < 0 else "與去年同期持平。"))
        if prev_month in revenue.index:
            prev_yoy = revenue.loc[prev_month, "月營收YoY"]
            if pd.notna(prev_yoy):
                diff = yoy - prev_yoy
                notes.append(f"**年增率動能**：較前一個月"
                             f"{'提高' if diff > 0 else '下降' if diff < 0 else '持平'}"
                             f" {abs(diff):.1f} 個百分點；"
                             "這是成長速度的變化，不等於營收絕對值同幅增減。")

    # 3月與12月均線變化：使用連續日曆月份，避免資料缺漏造成錯判。
    for col, label in [("MA3", "3 個月平均營收"), ("MA12", "12 個月平均營收")]:
        value = current[col]
        if pd.isna(value):
            notes.append(f"**{label}**：資料尚不足，無法判斷趨勢。")
            continue
        note = f"**{label}**：{number(value)} 億元"
        if prev_month in revenue.index:
            prev = revenue.loc[prev_month, col]
            delta = change(value, prev)
            if delta is not None:
                direction = "上升" if delta > 0 else "下降" if delta < 0 else "持平"
                note += f"，較上月{direction} {abs(delta):.1f}%"
        notes.append(note + "。")

    ma3, ma12 = current["MA3"], current["MA12"]
    rev = current["月營收"]
    if pd.notna(ma3):
        gap = change(rev, ma3)
        if gap is not None:
            notes.append(
                f"**當月營收相對短期均線**："
                f"{'高於' if gap > 0 else '低於' if gap < 0 else '等於'} MA3"
                f"{f' {abs(gap):.1f}%' if gap != 0 else ''}；"
                + ("近期單月表現高於近三個月平均。" if gap > 0 else
                   "近期單月表現低於近三個月平均。" if gap < 0 else
                   "與近三個月平均相當。")
            )
    if pd.notna(ma3) and pd.notna(ma12):
        spread = change(ma3, ma12)
        if spread is not None:
            notes.append(
                f"**短中期結構**：MA3 "
                f"{'高於' if spread > 0 else '低於' if spread < 0 else '等於'} MA12"
                f"{f' {abs(spread):.1f}%' if spread != 0 else ''}；"
                + ("近三個月平均營收高於近一年水準，短期營收動能相對較強。" if spread > 0 else
                   "近三個月平均營收低於近一年水準，短期營收動能相對較弱。" if spread < 0 else
                   "短期與全年平均營收接近。")
            )

    # 連續月數只檢查 MA3，且不得跨過缺失月份。
    if pd.notna(ma3):
        consecutive = 0
        direction = None
        cursor = month
        while consecutive < 12:
            previous = cursor - pd.DateOffset(months=1)
            if previous not in revenue.index:
                break
            now_val = revenue.loc[cursor, "MA3"]
            prev_val = revenue.loc[previous, "MA3"]
            if pd.isna(now_val) or pd.isna(prev_val) or now_val == prev_val:
                break
            step_direction = "上升" if now_val > prev_val else "下降"
            if direction is None:
                direction = step_direction
            elif direction != step_direction:
                break
            consecutive += 1
            cursor = previous
        if consecutive >= 2:
            suffix = "以上" if consecutive == 12 else ""
            notes.append(f"**短期趨勢持續性**：MA3 已連續 {consecutive} 個月{suffix}{direction}。")

    for note in notes:
        st.markdown("- " + note)

    st.caption("以上為依歷史營收數值自動產生的規則式分析，並非 AI 預測或投資建議。"
               "月營收及移動平均可能受季節性、工作天數、併購與認列時點影響；"
               "MA3／MA12 是落後指標，無法單獨判斷未來獲利或股價。")


def render_growth_driver_analysis(revenue):
    """以本年與去年相鄰月份的營收，拆解 YoY 變化的數學來源。"""
    st.markdown("#### 營收成長加速／減速原因分析")
    valid = revenue.dropna(subset=["月營收"]).sort_index()
    if valid.empty:
        st.info("尚無月營收資料可供分析。")
        return

    t = valid.index[-1]
    dates = [t - pd.DateOffset(months=n) for n in (0, 1, 12, 13)]
    if any(d not in revenue.index or pd.isna(revenue.loc[d, "月營收"])
           or revenue.loc[d, "月營收"] <= 0 for d in dates):
        st.info("需要本月、上月，以及去年對應的兩個月份均有正值營收，才能拆解年增率變化。")
        return

    a, b, c, d = [float(revenue.loc[date, "月營收"]) for date in dates]
    this_mom = (a / b - 1) * 100
    last_mom = (c / d - 1) * 100
    this_yoy = (a / c - 1) * 100
    prev_yoy = (b / d - 1) * 100
    yoy_delta = this_yoy - prev_yoy

    comparison = pd.DataFrame([
        {"比較月份": f"去年 {dates[3]:%Y/%m}", "月營收（億元）": d},
        {"比較月份": f"去年 {dates[2]:%Y/%m}", "月營收（億元）": c},
        {"比較月份": f"今年 {dates[1]:%Y/%m}", "月營收（億元）": b},
        {"比較月份": f"今年 {dates[0]:%Y/%m}", "月營收（億元）": a},
    ])
    with st.expander("查看今年與去年同期營收比較", expanded=False):
        st.dataframe(comparison.style.format({"月營收（億元）": "{:,.2f}"}),
                     use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("今年本月月增率", f"{this_mom:+.1f}%")
    c2.metric("去年同期月增率", f"{last_mom:+.1f}%")
    c3.metric("YoY 較上月變化", f"{yoy_delta:+.1f} 個百分點")

    # 使用對數成長率精確拆解：
    # log[(1+YoY_t)/(1+YoY_t-1)] = log(當年月增倍數) - log(去年同期月增倍數)。
    # 不將百分點直接相減當作精確貢獻，避免非線性誤差。
    import math
    current_effect = math.log(a / b)
    base_effect = -math.log(c / d)
    net_effect = current_effect + base_effect
    tol = 1e-10

    if abs(net_effect) < tol:
        st.markdown("**歸納：** YoY 與上月幾乎持平，當期變化與去年比較基期的影響大致抵銷。")
    else:
        accelerating = net_effect > 0
        direction = "加速" if accelerating else "放緩"
        st.markdown(f"**歸納：** 本月 YoY 為 **{this_yoy:+.1f}%**，上月為 **{prev_yoy:+.1f}%**，"
                    f"年增率較上月**{direction} {abs(yoy_delta):.1f} 個百分點**。")

        current_supports = current_effect * net_effect > tol
        base_supports = base_effect * net_effect > tol
        if current_supports and base_supports:
            interpretation = "今年營收的月度變化與去年比較基期變化，都朝同一方向推動 YoY。"
        elif current_supports and not base_supports:
            interpretation = "今年營收的月度變化是主要推動因素，去年基期變化則部分抵銷其影響。"
        elif base_supports and not current_supports:
            interpretation = "去年比較基期變化是主要推動因素，今年營收的月度變化則部分抵銷其影響。"
        else:
            interpretation = "當期營收與比較基期的影響互相抵銷，淨變化很小。"
        st.markdown(f"**數學拆解：** {interpretation}")

    current_phrase = ("增加" if this_mom > 0 else "減少" if this_mom < 0 else "持平")
    base_phrase = ("墊高" if last_mom > 0 else "降低" if last_mom < 0 else "持平")
    st.markdown(f"- **今年營收變化：** 本月相較上月{current_phrase} **{abs(this_mom):.1f}%**。")
    st.markdown(f"- **去年比較基期：** 去年同期月營收相較其前月"
                f"{'增加' if last_mom > 0 else '減少' if last_mom < 0 else '持平'} "
                f"**{abs(last_mom):.1f}%**，使本月的年增率比較基期相對{base_phrase}。")
    st.caption("判讀依據：YoY 的相鄰月變化，取決於今年月營收變動倍數與去年同期月營收變動倍數的比值。"
               "此處是數學上的基期／當期拆解，不代表已證實需求、產品或產業面的實際原因；"
               "工作天數、季節性、併購與營收認列時點仍需搭配公告查證。")


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
    (show_ma12, "MA12", "#16a34a", 3.0),
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
render_revenue_insights(df)
render_growth_driver_analysis(df)


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
