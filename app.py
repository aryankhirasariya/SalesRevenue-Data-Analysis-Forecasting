import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Sales & Revenue Analytics",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    h1, h2, h3 {
        color: #ffffff !important;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 16px;
        margin-bottom: 25px;
    }
    [data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 15px 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #1e293b;
        border-radius: 15px !important;
        padding: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    [data-testid="stVerticalBlockBorderWrapper"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #0f172a;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<h1 style='text-align:center;'>💹 Sales & Revenue Analytics Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analyze historical sales trends and forecast future revenue using Python</div>", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.info("This dashboard analyzes sales data, visualizes trends, and forecasts future revenue using a Linear Regression model.")
    st.markdown("### 🛠️ Tech Stack")
    st.write("- Python\n- Pandas\n- Plotly\n- Scikit-learn")
    st.markdown("### 📂 Data Source")
    uploaded_file = st.file_uploader("Upload your Sales CSV", type=["csv"])

# ---------- LOAD DATA ----------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce", dayfirst=True)
    df.dropna(subset=["Order Date", "Sales"], inplace=True)
    return df

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    st.warning("👆 Upload the Superstore Sales CSV from the sidebar to get started.")
    st.stop()

# ---------- KPIs ----------
total_sales = df["Sales"].sum()
total_orders = df["Order ID"].nunique() if "Order ID" in df.columns else len(df)
avg_order_value = total_sales / total_orders if total_orders else 0
top_region = df.groupby("Region")["Sales"].sum().idxmax() if "Region" in df.columns else "N/A"

col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Total Sales", f"${total_sales:,.0f}")
col2.metric("🧾 Total Orders", f"{total_orders:,}")
col3.metric("📦 Avg Order Value", f"${avg_order_value:,.2f}")
col4.metric("🌍 Top Region", top_region)

st.write("")

# ---------- MONTHLY SALES TREND ----------
with st.container(border=True):
    st.markdown("#### 📈 Monthly Sales Trend")
    monthly_sales = df.set_index("Order Date").resample("ME")["Sales"].sum().reset_index()
    fig = px.line(
        monthly_sales, x="Order Date", y="Sales",
        markers=True, template="plotly_dark",
        color_discrete_sequence=["#6366f1"]
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------- CATEGORY & REGION BREAKDOWN ----------
col_a, col_b = st.columns(2)

with col_a:
    with st.container(border=True):
        st.markdown("#### 🏷️ Sales by Category")
        if "Category" in df.columns:
            cat_sales = df.groupby("Category")["Sales"].sum().reset_index()
            fig2 = px.pie(
                cat_sales, names="Category", values="Sales",
                hole=0.5, template="plotly_dark",
                color_discrete_sequence=px.colors.sequential.Purples_r
            )
            fig2.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig2, use_container_width=True)

with col_b:
    with st.container(border=True):
        st.markdown("#### 🌎 Sales by Region")
        if "Region" in df.columns:
            reg_sales = df.groupby("Region")["Sales"].sum().reset_index()
            fig3 = px.bar(
                reg_sales, x="Region", y="Sales",
                template="plotly_dark", color="Region",
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig3.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False
            )
            st.plotly_chart(fig3, use_container_width=True)

# ---------- TOP PRODUCTS ----------
with st.container(border=True):
    st.markdown("#### 🏆 Top 10 Products by Sales")
    if "Product Name" in df.columns:
        top_products = df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(10).reset_index()
        fig4 = px.bar(
            top_products, x="Sales", y="Product Name",
            orientation="h", template="plotly_dark",
            color="Sales", color_continuous_scale="Purples"
        )
        fig4.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig4, use_container_width=True)

# ---------- FORECASTING ----------
with st.container(border=True):
    st.markdown("#### 🔮 Sales Forecast (Next N Months)")

    months_ahead = st.slider("Select number of months to forecast", 1, 12, 6)

    # Prepare monthly data for regression
    ts = monthly_sales.copy()
    ts["MonthIndex"] = np.arange(len(ts))

    X = ts[["MonthIndex"]]
    y = ts["Sales"]

    model = LinearRegression()
    model.fit(X, y)

    future_index = np.arange(len(ts), len(ts) + months_ahead).reshape(-1, 1)
    future_preds = model.predict(future_index)

    future_dates = pd.date_range(
    start=ts["Order Date"].iloc[-1] + pd.offsets.MonthEnd(1),
    periods=months_ahead, freq="ME"
        )  

    forecast_df = pd.DataFrame({"Order Date": future_dates, "Sales": future_preds, "Type": "Forecast"})
    actual_df = ts[["Order Date", "Sales"]].copy()
    actual_df["Type"] = "Actual"

    combined = pd.concat([actual_df, forecast_df])

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=actual_df["Order Date"], y=actual_df["Sales"],
        mode="lines+markers", name="Actual Sales",
        line=dict(color="#6366f1", width=3)
    ))
    fig5.add_trace(go.Scatter(
        x=forecast_df["Order Date"], y=forecast_df["Sales"],
        mode="lines+markers", name="Forecasted Sales",
        line=dict(color="#f59e0b", width=3, dash="dash")
    ))
    fig5.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.caption("📌 Forecast generated using Linear Regression on monthly aggregated sales data. For production use, consider ARIMA, Prophet, or SARIMA for better seasonality handling.")

# ---------- RAW DATA ----------
with st.expander("🔍 View Raw Data"):
    st.dataframe(df, use_container_width=True)