from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UI_DIR = PROJECT_ROOT / "artifacts" / "ui"


@st.cache_data(show_spinner=False)
def load_ui_bundle(ui_dir: Path) -> Tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    topics = pd.read_parquet(ui_dir / "topics.parquet")
    topics_info = pd.read_parquet(ui_dir / "topics_info.parquet")
    trends = pd.read_parquet(ui_dir / "trends.parquet")
    examples = pd.read_parquet(ui_dir / "topic_examples.parquet")

    meta_path = ui_dir / "meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Normalize topics_info to stable names
    if "Topic" in topics_info.columns and "topic_id" not in topics_info.columns:
        topics_info = topics_info.rename(columns={"Topic": "topic_id"})
    if "Name" not in topics_info.columns:
        topics_info["Name"] = topics_info.get("Representation", "")
    if "keywords" not in topics_info.columns:
        topics_info["keywords"] = topics_info.get("Representation", "")

    # Parse datetimes
    topics["created_at"] = pd.to_datetime(topics.get("created_at"), utc=True, errors="coerce")
    trends["time_bin"] = pd.to_datetime(trends.get("time_bin"), utc=True, errors="coerce")
    examples["created_at"] = pd.to_datetime(examples.get("created_at"), utc=True, errors="coerce")

    # Ensure numeric types
    topics["topic_id"] = pd.to_numeric(topics["topic_id"], errors="coerce").fillna(-1).astype(int)
    topics["topic_prob"] = pd.to_numeric(topics["topic_prob"], errors="coerce")

    topics_info["topic_id"] = pd.to_numeric(topics_info["topic_id"], errors="coerce").fillna(-1).astype(int)
    if "Count" in topics_info.columns:
        topics_info["Count"] = pd.to_numeric(topics_info["Count"], errors="coerce").fillna(0).astype(int)

    trends["topic_id"] = pd.to_numeric(trends["topic_id"], errors="coerce").fillna(-1).astype(int)
    for c in ["share", "growth", "n_posts", "avg_prob"]:
        if c in trends.columns:
            trends[c] = pd.to_numeric(trends[c], errors="coerce")

    examples["topic_id"] = pd.to_numeric(examples["topic_id"], errors="coerce").fillna(-1).astype(int)
    if "topic_prob" in examples.columns:
        examples["topic_prob"] = pd.to_numeric(examples["topic_prob"], errors="coerce")

    # Safety: ensure required cols exist
    for c in ["post_id", "url"]:
        if c not in topics.columns:
            topics[c] = None
    for c in ["url", "text", "topic_prob"]:
        if c not in examples.columns:
            examples[c] = None

    return meta, topics, topics_info, trends, examples


def compute_delta_share(trends: pd.DataFrame, start, end) -> pd.DataFrame:
    sub = trends[(trends["time_bin"] >= start) & (trends["time_bin"] <= end)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["topic_id", "share_first", "share_last", "delta_share", "n_last"])

    sub = sub.sort_values(["topic_id", "time_bin"])
    first = sub.groupby("topic_id", as_index=False).first()[["topic_id", "share"]].rename(columns={"share": "share_first"})
    last = sub.groupby("topic_id", as_index=False).last()[["topic_id", "share", "n_posts"]].rename(
        columns={"share": "share_last", "n_posts": "n_last"}
    )
    out = first.merge(last, on="topic_id", how="inner")
    out["delta_share"] = out["share_last"] - out["share_first"]
    return out


def make_heatmap_matrix(trends_f: pd.DataFrame, topic_ids: List[int], metric: str = "share") -> pd.DataFrame:
    sub = trends_f[trends_f["topic_id"].isin(topic_ids)].copy()
    if sub.empty:
        return pd.DataFrame()

    mat = sub.pivot_table(index="topic_id", columns="time_bin", values="share", aggfunc="mean").fillna(0.0)

    if metric == "z-score":
        mu = mat.mean(axis=1)
        sigma = mat.std(axis=1).replace(0, np.nan)
        mat = (mat.sub(mu, axis=0)).div(sigma, axis=0).fillna(0.0)

    if mat.shape[1] > 0:
        last_col = mat.columns.max()
        mat = mat.sort_values(last_col, ascending=False)

    return mat


def build_report_md(meta: dict, filters: dict, rank_df: pd.DataFrame) -> str:
    prof = meta.get("profile_name", "unknown")
    stats = meta.get("stats", {})
    prob = stats.get("topic_prob", {})

    lines = []
    lines.append("# Social Trends Lab — Report")
    lines.append("")
    lines.append(f"**Profile:** `{prof}`")
    lines.append("")
    lines.append("## Dataset / Result snapshot")
    lines.append(f"- n_docs: **{stats.get('n_docs', 'n/a')}**")
    lines.append(f"- noise_share: **{stats.get('noise_share', 'n/a')}**")
    lines.append(f"- median(topic_prob): **{prob.get('median', 'n/a')}**")
    lines.append(f"- share(topic_prob≥0.4): **{prob.get('p_ge_0_4', 'n/a')}**")
    lines.append("")
    lines.append("## UI filters used for this report")
    for k, v in filters.items():
        lines.append(f"- {k}: **{v}**")
    lines.append("")
    lines.append("## Top topics (current view)")
    if rank_df is None or rank_df.empty:
        lines.append("_No topics under current filters._")
    else:
        cols = [c for c in ["topic_id", "share", "delta_share", "growth", "n_posts", "avg_prob", "keywords"] if c in rank_df.columns]
        lines.append("")
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, r in rank_df.head(20).iterrows():
            row = []
            for c in cols:
                val = r.get(c)
                if isinstance(val, float):
                    row.append(f"{val:.6f}")
                else:
                    row.append(str(val))
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def _kpi_block(meta: dict, topics: pd.DataFrame) -> None:
    stats = meta.get("stats", {})
    prob = stats.get("topic_prob", {})

    # fallback if meta not present
    n_docs = stats.get("n_docs", len(topics))
    noise_share = stats.get("noise_share", float((topics["topic_id"] == -1).mean()) if len(topics) else 0.0)
    median_prob = prob.get("median", float(pd.to_numeric(topics["topic_prob"], errors="coerce").median()) if len(topics) else None)
    p_ge_04 = prob.get("p_ge_0_4", float((pd.to_numeric(topics["topic_prob"], errors="coerce") >= 0.4).mean()) if len(topics) else None)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Docs", f"{n_docs:,}")
    c2.metric("Noise share", f"{float(noise_share):.3f}")
    c3.metric("Median prob", f"{float(median_prob):.3f}" if median_prob is not None else "n/a")
    c4.metric("Prob ≥ 0.4", f"{float(p_ge_04):.3f}" if p_ge_04 is not None else "n/a")


def main() -> None:
    st.set_page_config(page_title="Social Trends Lab", layout="wide")

    st.title("Social Trends Lab")
    st.caption("BERTopic → trends → интерактивная аналитика. Данные: Reddit submissions (HF).")

    if not UI_DIR.exists():
        st.error(f"UI bundle not found: {UI_DIR}")
        st.stop()

    meta, topics, topics_info, trends, examples = load_ui_bundle(UI_DIR)

    # ===== Sidebar filters =====
    st.sidebar.header("Фильтры")

    presets = {
        "Showcase (2018–2020)": ("2018-01-01", "2020-12-31"),
        "Recent (2019–2022)": ("2019-01-01", "2022-12-31"),
        "All time": (None, None),
    }
    preset_name = st.sidebar.selectbox("Preset периода", list(presets.keys()), index=0)

    min_date = trends["time_bin"].min()
    max_date = trends["time_bin"].max()
    if pd.isna(min_date) or pd.isna(max_date):
        st.error("Bad trends.time_bin parsing. Check artifacts/ui/trends.parquet")
        st.stop()

    p_start, p_end = presets[preset_name]
    if p_start and p_end:
        start_default = max(pd.Timestamp(p_start).tz_localize("UTC"), min_date)
        end_default = min(pd.Timestamp(p_end).tz_localize("UTC"), max_date)
    else:
        start_default, end_default = min_date, max_date

    date_range = st.sidebar.date_input(
        "Период (UTC)",
        value=(start_default.date(), end_default.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date, end_date = start_default.date(), end_default.date()

    start = pd.Timestamp(start_date).tz_localize("UTC")
    end = pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    include_noise = st.sidebar.checkbox("Include noise (-1)", value=False)
    min_topic_prob = st.sidebar.slider("min topic_prob", 0.0, 1.0, 0.30, step=0.05)

    min_topic_size = st.sidebar.slider("Минимальный размер темы (Count)", 10, 5000, 200, step=10)
    top_k = st.sidebar.slider("Топ тем", 5, 50, 15, step=1)

    rank_mode = st.sidebar.selectbox(
        "Ранжирование трендов",
        ["Δshare (популярность)", "growth (WoW)", "share (последний бин)"],
        index=0,
    )

    search = st.sidebar.text_input("Поиск по keywords/Name", value="")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Визуализации")
    show_heatmap = st.sidebar.checkbox("Показать heatmap Topics×Time", value=True)
    heatmap_metric = st.sidebar.selectbox("Метрика heatmap", ["share", "z-score"], index=0)
    heatmap_topics_n = st.sidebar.slider("Heatmap: число тем", 10, 100, min(40, top_k), step=5)

    show_compare = st.sidebar.checkbox("Сравнить несколько тем (линии)", value=True)
    compare_max = st.sidebar.slider("Compare: максимум тем", 2, 12, 6, step=1)

    # ===== Filter topics_info =====
    ti = topics_info.copy()
    if "Count" in ti.columns:
        ti = ti[ti["Count"] >= min_topic_size].copy()
    if not include_noise:
        ti = ti[ti["topic_id"] != -1].copy()

    if search.strip():
        s = search.strip().lower()
        ti = ti[
            ti["keywords"].fillna("").astype(str).str.lower().str.contains(s)
            | ti["Name"].fillna("").astype(str).str.lower().str.contains(s)
        ].copy()

    allowed_topics = set(ti["topic_id"].astype(int).tolist())

    # ===== Filter trends/topics/examples by date & allowed topics =====
    trends_f = trends[trends["topic_id"].isin(allowed_topics)].copy()
    trends_f = trends_f[(trends_f["time_bin"] >= start) & (trends_f["time_bin"] <= end)].copy()

    topics_f = topics[topics["topic_id"].isin(allowed_topics)].copy()
    topics_f = topics_f[(topics_f["created_at"] >= start) & (topics_f["created_at"] <= end)].copy()
    topics_f = topics_f[topics_f["topic_prob"].fillna(-1) >= float(min_topic_prob)].copy()

    examples_f = examples[examples["topic_id"].isin(allowed_topics)].copy()
    examples_f = examples_f[(examples_f["created_at"] >= start) & (examples_f["created_at"] <= end)].copy()
    examples_f = examples_f[examples_f["topic_prob"].fillna(-1) >= float(min_topic_prob)].copy()

    if trends_f.empty:
        st.warning("По текущим фильтрам тренды пустые. Расширь период или снизь min_topic_size.")
        st.stop()

    last_bin = trends_f["time_bin"].max()
    last_slice = trends_f[trends_f["time_bin"] == last_bin][["topic_id", "share", "growth", "n_posts", "avg_prob"]].copy()

    delta = compute_delta_share(trends_f, start, end)

    rank = last_slice.merge(delta[["topic_id", "delta_share"]], on="topic_id", how="left")
    rank = rank.merge(ti[["topic_id", "Name", "keywords"]], on="topic_id", how="left")
    rank["delta_share"] = rank["delta_share"].fillna(0.0)

    if rank_mode.startswith("Δshare"):
        rank = rank.sort_values("delta_share", ascending=False)
    elif rank_mode.startswith("growth"):
        rank = rank.sort_values("growth", ascending=False)
    else:
        rank = rank.sort_values("share", ascending=False)

    rank = rank.head(top_k).reset_index(drop=True)

    # ===== Downloads =====
    st.sidebar.markdown("---")
    trends_csv = trends_f.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        "Download trends_filtered.csv",
        data=trends_csv,
        file_name="trends_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

    rank_csv = rank.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        "Download rank_table.csv",
        data=rank_csv,
        file_name="rank_table.csv",
        mime="text/csv",
        use_container_width=True,
    )

    report_filters = {
        "preset": preset_name,
        "period": f"{start_date}..{end_date} UTC",
        "include_noise": include_noise,
        "min_topic_prob": float(min_topic_prob),
        "min_topic_size": int(min_topic_size),
        "rank_mode": rank_mode,
        "search": search.strip(),
        "heatmap_metric": heatmap_metric if show_heatmap else "off",
    }
    report_md = build_report_md(meta, report_filters, rank)
    st.sidebar.download_button(
        "Download report.md",
        data=report_md.encode("utf-8"),
        file_name="report.md",
        mime="text/markdown",
        use_container_width=True,
    )

    # ===== Main tabs =====
    tab_overview, tab_inspector, tab_trends = st.tabs(["Overview", "Topic Inspector", "Trends"])

    with tab_overview:
        st.subheader("Overview")
        _kpi_block(meta, topics)

        st.markdown("### Trending now (таблица)")
        st.caption(f"Последний бин: {last_bin.date()} | ранжирование: {rank_mode}")

        show_cols = ["topic_id", "share", "delta_share", "growth", "n_posts", "avg_prob", "keywords"]
        rank_show = rank.copy()
        for c in ["share", "delta_share", "growth", "avg_prob"]:
            if c in rank_show.columns:
                rank_show[c] = pd.to_numeric(rank_show[c], errors="coerce")
        st.dataframe(rank_show[show_cols], use_container_width=True, height=420)

        st.markdown("### Top movers (карточки)")
        cards = rank.head(min(6, len(rank))).copy()
        for _, r in cards.iterrows():
            tid = int(r["topic_id"])
            st.markdown(
                f"**Topic {tid}** — Δshare={float(r['delta_share']):+.4f}, share={float(r['share']):.4f}, "
                f"growth={float(r['growth']):.3f} (если NaN — мало данных)"
            )
            st.write(f"Keywords: {r.get('keywords','')}")
            ex1 = examples_f[examples_f["topic_id"] == tid].sort_values("topic_prob", ascending=False).head(1)
            if len(ex1):
                st.write(f"Example: {ex1.iloc[0]['text']}")
            st.divider()

    with tab_inspector:
        st.subheader("Topic Inspector")

        if rank.empty:
            st.info("Нет тем по текущим фильтрам.")
            st.stop()

        selected = st.selectbox("Topic", options=rank["topic_id"].astype(int).tolist(), index=0)
        row = rank[rank["topic_id"] == selected].iloc[0]

        st.write(f"**Keywords:** {row.get('keywords','')}")
        st.write(
            f"share={row['share']:.4f} | Δshare={row['delta_share']:+.4f} | "
            f"growth={row['growth'] if pd.notna(row['growth']) else np.nan:.3f} | n_posts(last)={int(row['n_posts'])}"
        )

        tsub = trends_f[trends_f["topic_id"] == selected].sort_values("time_bin")
        fig = px.line(tsub, x="time_bin", y="share", markers=True, title="Share over time")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Examples")
        ex = examples_f[examples_f["topic_id"] == selected].copy()
        ex = ex.sort_values("topic_prob", ascending=False).head(30)

        if ex.empty:
            st.info("Нет примеров по текущим фильтрам.")
        else:
            for r in ex.itertuples(index=False):
                txt = str(r.text)
                txt = (txt[:420] + " ...") if len(txt) > 420 else txt
                st.write(f"**p={r.topic_prob:.3f}** — {txt}")
                if isinstance(r.url, str) and r.url:
                    st.link_button("Open", r.url)
                st.divider()

    with tab_trends:
        st.subheader("Trends")

        if show_heatmap:
            st.markdown("### Heatmap Topics × Time")
            top_topics_for_heatmap = rank["topic_id"].astype(int).head(heatmap_topics_n).tolist()
            mat = make_heatmap_matrix(
                trends_f,
                top_topics_for_heatmap,
                metric=("z-score" if heatmap_metric == "z-score" else "share"),
            )

            if mat.empty:
                st.info("Heatmap пустой по текущим фильтрам.")
            else:
                fig_hm = px.imshow(
                    mat,
                    aspect="auto",
                    labels=dict(x="time_bin", y="topic_id", color=heatmap_metric),
                )
                fig_hm.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_hm, use_container_width=True)

                hm_csv = mat.reset_index().to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download heatmap_matrix.csv",
                    data=hm_csv,
                    file_name="heatmap_matrix.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        if show_compare:
            st.markdown("### Compare topics (multi-line)")
            options = rank["topic_id"].astype(int).tolist()
            default_sel = options[: min(compare_max, len(options))]
            chosen = st.multiselect("Выбери темы", options=options, default=default_sel)

            if chosen:
                sub = trends_f[trends_f["topic_id"].isin(chosen)].copy().sort_values("time_bin")
                fig_cmp = px.line(sub, x="time_bin", y="share", color="topic_id", title="Share comparison")
                fig_cmp.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_cmp, use_container_width=True)


if __name__ == "__main__":
    main()
