"""面积比 R² 柱状图，单模型与多模型共用相同坐标和图例。"""
import altair as alt
import streamlit as st

MODEL_COLORS = {
    "ANN": "#F72585",       # 亮粉
    "XGBOOST": "#FF8500",   # 橙色
    "RF": "#00B894",        # 翠绿
    "GPR": "#00B4D8",       # 青蓝
    "SVR": "#7B2CF4",       # 亮紫
    "PR": "#F4C430",        # 金黄
}


def render_r2_chart(metrics):
    data = metrics[["Model", "R2"]].copy()
    data["Target"] = "Area rate"
    multi_model = len(data) > 1
    names = [name for name in MODEL_COLORS if name in data["Model"].values]
    color = alt.Color(
        "Model:N" if multi_model else "Target:N", title=None,
        scale=alt.Scale(
            domain=names if multi_model else ["Area rate"],
            range=[MODEL_COLORS[name] for name in names] if multi_model else ["#4F61A6"],
        ),
        legend=alt.Legend(orient="top", symbolType="square",
                          labelFontSize=14, labelFontWeight="bold"),
    )
    chart = (
        alt.Chart(data)
        .mark_bar(size=42, clip=True)
        .encode(
            x=alt.X("Model:N", sort=data["Model"].tolist(),
                    title="Model", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("R2:Q", title="R²",
                    scale=alt.Scale(domain=[-1, 1], nice=False, zero=True),
                    axis=alt.Axis(values=[-1, -0.5, 0, 0.5, 1], format=".1f")),
            color=color,
            tooltip=[alt.Tooltip("Model:N", title="模型"),
                     alt.Tooltip("R2:Q", title="R²", format=".4f")],
        )
        .properties(height=330)
        .configure_axis(grid=False, labelFontSize=12, titleFontSize=14)
        .configure_view(stroke="#333333")
    )
    st.altair_chart(chart, use_container_width=True)
    outside = data.loc[(data["R2"] < -1) | (data["R2"] > 1)]
    if not outside.empty:
        details = "、".join(f"{row.Model}: {row.R2:.4f}" for row in outside.itertuples())
        st.caption(f"以下模型的 R² 超出本图 −1～1 显示范围：{details}。完整数值见上方表格。")
