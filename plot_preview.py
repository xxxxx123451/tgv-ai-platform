"""仅控制网页图片预览大小，不改变投稿图片的导出尺寸。"""
import base64

import streamlit as st


def render_plot_preview(png, *, key, base_width=680):
    zoom = st.slider(
        "预览大小（%）", min_value=40, max_value=160, value=100,
        step=10, key=f"{key}_zoom",
        help="向左缩小、向右放大；图片始终适配当前窗口，不影响下载图片。",
    )
    width = round(base_width * zoom / 100)
    # 使用视口高度和容器宽度双重约束，避免单模型图片撑满整页。
    height_vh = min(85, round(60 * zoom / 100))
    encoded = base64.b64encode(png).decode("ascii")
    st.markdown(
        '<div style="width:100%;display:flex;justify-content:center;'
        'align-items:center;padding:12px 0;">'
        f'<img src="data:image/png;base64,{encoded}" alt="模型评估图预览" '
        f'style="display:block;width:{width}px;max-width:100%;'
        f'height:auto;max-height:{height_vh}vh;object-fit:contain;'
        'border-radius:8px;" /></div>',
        unsafe_allow_html=True,
    )
