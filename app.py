import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------
# HIDE STREAMLIT DEFAULT HEADER SPACE
# -------------------------------------------------
st.markdown("""
<style>
header {visibility: hidden;}
footer {visibility: hidden;}
.block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# CUSTOM DASHBOARD STYLING
# -------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.dashboard {
    background: linear-gradient(135deg, #020617, #0f172a);
    padding: 30px;
    border-radius: 16px;
}
.title {
    font-size: 42px;
    font-weight: 700;
    color: #436175;
}
.subtitle {
    color: #94a3b8;
    font-size: 18px;
    margin-bottom: 25px;
}
.metric-box {
    background-color: #020617;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: #e5e7eb;
}
.footer {
    text-align: right;
    color: #64748b;
    margin-top: 40px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
data = pd.read_csv("sales_data.csv")
data["month_index"] = np.arange(1, len(data) + 1)

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.title("🔍 Filters")

month_range = st.sidebar.slider(
    "Select Month Range",
    min_value=1,
    max_value=len(data),
    value=(1, len(data))
)

sales_range = st.sidebar.slider(
    "Select Sales Range",
    min_value=int(data.sales.min()),
    max_value=int(data.sales.max()),
    value=(int(data.sales.min()), int(data.sales.max()))
)

# Apply filters
filtered_data = data[
    (data["month_index"] >= month_range[0]) &
    (data["month_index"] <= month_range[1]) &
    (data["sales"] >= sales_range[0]) &
    (data["sales"] <= sales_range[1])
]

# -------------------------------------------------
# DASHBOARD START
# -------------------------------------------------
st.markdown('<div class="dashboard">', unsafe_allow_html=True)

st.markdown('<div class="title">📊 Sales Analytics & Forecasting Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Interactive Data Science Web Application</div>', unsafe_allow_html=True)

# -------------------------------------------------
# DATA TABLE
# -------------------------------------------------
st.markdown("### 📋 Filtered Sales Data")
st.dataframe(
    filtered_data[["month", "sales"]],
    use_container_width=True,
    height=260
)

# -------------------------------------------------
# METRICS
# -------------------------------------------------
st.markdown("### 📈 Key Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-box">
        <h2>₹ {int(filtered_data.sales.mean()) if not filtered_data.empty else 0}</h2>
        <p>Average Sales</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-box">
        <h2>₹ {filtered_data.sales.max() if not filtered_data.empty else 0}</h2>
        <p>Highest Sales</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-box">
        <h2>₹ {filtered_data.sales.min() if not filtered_data.empty else 0}</h2>
        <p>Lowest Sales</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# TREND LINE GRAPH (KEPT)
# -------------------------------------------------
st.markdown("### 📉 Monthly Sales Trend (Line Chart)")
fig_line, ax_line = plt.subplots()
ax_line.plot(filtered_data["month"], filtered_data["sales"], marker="o")
ax_line.set_xlabel("Month")
ax_line.set_ylabel("Sales")
ax_line.set_title("Sales Trend Over Time")
st.pyplot(fig_line)

# -------------------------------------------------
# BAR CHART
# -------------------------------------------------
st.markdown("### 📊 Monthly Sales Comparison (Bar Chart)")
fig_bar, ax_bar = plt.subplots()
ax_bar.bar(filtered_data["month"], filtered_data["sales"])
ax_bar.set_xlabel("Month")
ax_bar.set_ylabel("Sales")
ax_bar.set_title("Sales by Month")
st.pyplot(fig_bar)

# -------------------------------------------------
# HISTOGRAM
# -------------------------------------------------
st.markdown("### 📊 Sales Distribution (Histogram)")
fig_hist, ax_hist = plt.subplots()
ax_hist.hist(filtered_data["sales"], bins=5)
ax_hist.set_xlabel("Sales")
ax_hist.set_ylabel("Frequency")
ax_hist.set_title("Sales Frequency Distribution")
st.pyplot(fig_hist)

# -------------------------------------------------
# MACHINE LEARNING MODEL
# -------------------------------------------------
X = data[["month_index"]]
y = data["sales"]

model = LinearRegression()
model.fit(X, y)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
st.markdown("### 🔮 Sales Forecast")

future_month = st.slider("Select Future Month (13 = Next Month)", 13, 24, 13)
prediction = model.predict([[future_month]])

st.success(f"Predicted Sales: ₹ {int(prediction[0])}")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("""
<div class="footer">
    Mohitha Nandini | 23B01A4582
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
