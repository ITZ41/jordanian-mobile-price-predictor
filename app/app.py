"""Streamlit dashboard — Jordanian Mobile Price Predictor.

5 tabs:
  1. Price Estimate  — Predict price, confidence interval, deal score, similar listings
  2. Market Analysis  — Interactive Plotly charts
  3. Trade-In Calculator — Upgrade options
  4. Brand Comparison — Side-by-side brand analysis
  5. My Listing Analyzer — Paste raw Arabic title, auto-extract + predict
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.express as px

from src.paths import MODEL_PATH, METRICS_PATH, CLEANED_DATA_PATH
from src.cleaning import extract_series, normalize_condition, normalize_arabic, extract_storage_gb

# ── Translations ──
TRANSLATIONS = {
    "en": {
        "title": "📈 Mobile Price Engine",
        "subtitle": "Jordanian Secondary Mobile Market Analyzer & Predictor",
        "device_specs": "Device Specifications",
        "brand": "Brand",
        "series": "Series",
        "storage": "Storage (GB)",
        "condition": "Condition",
        "asking_price": "Asking Price",
        "seller_price": "Seller's Price (JOD)",
        "premium_features": "Premium Features",
        "device_attributes": "Device Attributes",
        "warranty": "Official Warranty (كفالة)",
        "sealed": "New in Box (Sealed/كرتون)",
        "calculate": "Calculate Value",
        "predicted_value": "Predicted Market Value",
        "price_range": "Typical Price Range (68% confidence)",
        "deal_score": "Deal Score",
        "enter_asking": "Enter an asking price to evaluate",
        "comparable": "Comparable Listings",
        "no_comparable": "No direct comparable listings found in this price range.",
        "adjust_sidebar": "Adjust the sidebar parameters and click 'Calculate Value' to see the estimate.",
        "market_trends": "Market Trends",
        "listings_by_series": "Listings by Series",
        "price_dist": "Price Distribution",
        "price_by_condition": "Price Range by Condition",
        "trade_in_title": "🔄 Trade-In Calculator",
        "trade_in_desc": "Estimate what you can upgrade to by trading in your current phone.",
        "current_phone": "Your Current Phone",
        "upgrade_budget": "Your Upgrade Budget",
        "extra_budget": "Additional Budget (JOD)",
        "calc_trade": "Calculate Trade-In Options",
        "trade_in_value": "Your Trade-In Value",
        "upgrade_options": "Upgrade Options Within Budget",
        "no_upgrades": "No upgrade options found within your budget. Try increasing your budget.",
        "compare_title": "⚖️ Brand Value Comparison",
        "compare_desc": "Compare how different brands hold their value in the Jordanian market.",
        "compare_btn": "Compare Brands",
        "lang_label": "🌐 Language",
        "storage_all": "All",
        "market_overview": "Market Overview",
        "avg_price": "Avg Price",
        "median_price": "Median Price",
        "listings_count": "Listings",
        "price_by_storage": "Price by Storage & Condition",
        "all_listings": "All Listings",
        "no_listings": "No listings found for this brand and series.",
        "select_overview": "Select a brand and series to see market overview.",
        # New strings for tab 5
        "analyzer_title": "🔍 My Listing Analyzer",
        "analyzer_desc": "Paste a raw Arabic listing title and the app will automatically extract brand, series, storage, and condition — then predict the price.",
        "analyzer_input_label": "Paste listing title (Arabic)",
        "analyzer_input_placeholder": "مثال: ايفون 15 برو ماكس 256 مستعمل ممتاز بدل",
        "analyzer_analyze_btn": "Analyze Listing",
        "analyzer_extracted": "Extracted Information",
        "analyzer_brand": "Brand",
        "analyzer_series": "Series",
        "analyzer_storage": "Storage",
        "analyzer_condition": "Condition",
        "analyzer_predicted": "Predicted Price",
        "analyzer_range": "Price Range",
        "analyzer_deal": "Deal Score",
        "analyzer_no_title": "Please enter a listing title to analyze.",
        "similar_listings": "Similar Listings",
        "no_similar": "No similar listings found in dataset.",
        "confidence_interval": "Confidence Interval",
    },
    "ar": {
        "title": "📈 محرك أسعار الهواتف",
        "subtitle": "محلل ومتنبئ سوق الهواتف المستعملة الأردني",
        "device_specs": "مواصفات الجهاز",
        "brand": "العلامة التجارية",
        "series": "الفئة",
        "storage": "سعة التخزين (جيجابايت)",
        "condition": "الحالة",
        "asking_price": "السعر المطلوب",
        "seller_price": "سعر البائع (دينار)",
        "premium_features": "المزايا المميزة",
        "device_attributes": "صفات الجهاز",
        "warranty": "كفالة رسمية",
        "sealed": "جديد بالكرتون (مسكر)",
        "calculate": "احسب القيمة",
        "predicted_value": "القيمة السوقية المتنبأة",
        "price_range": "نطاق السعر المعتاد (ثقة 68%)",
        "deal_score": "معدل الصفقة",
        "enter_asking": "أدخل السعر المطلوب للتقييم",
        "comparable": "إعلانات مشابهة",
        "no_comparable": "لم يتم العثور على إعلانات مشابهة في هذا النطاق السعري.",
        "adjust_sidebar": "اضبط المعلمات في الشريط الجانبي واضغط 'احسب القيمة' لرؤية التقدير.",
        "market_trends": "اتجاهات السوق",
        "listings_by_series": "الإعلانات حسب الفئة",
        "price_dist": "توزيع الأسعار",
        "price_by_condition": "نطاق السعر حسب الحالة",
        "trade_in_title": "🔄 حاسبة المقايضة",
        "trade_in_desc": "قدر ما يمكنك الترقية إليه بمقايضة هاتفك الحالي.",
        "current_phone": "هاتفك الحالي",
        "upgrade_budget": "ميزانية الترقية",
        "extra_budget": "الميزانية الإضافية (دينار)",
        "calc_trade": "احسب خيارات المقايضة",
        "trade_in_value": "قيمة المقايضة",
        "upgrade_options": "خيارات الترقية ضمن الميزانية",
        "no_upgrades": "لم يتم العثور على خيارات ترقية ضمن ميزانيتك. حاول زيادة الميزانية.",
        "compare_title": "⚖️ مقارنة العلامات التجارية",
        "compare_desc": "قارن كيف تحتفظ العلامات المختلفة بقيمتها في السوق الأردني.",
        "compare_btn": "قارن العلامات",
        "lang_label": "🌐 اللغة",
        "storage_all": "الكل",
        "market_overview": "نظره عامه على السوق",
        "avg_price": "متوسط السعر",
        "median_price": "السعر الوسط",
        "listings_count": "الإعلانات",
        "price_by_storage": "السعر حسب التخزين والحاله",
        "all_listings": "جميع الإعلانات",
        "no_listings": "لم يتم العثور على إعلانات لهذه العلامة والفئة.",
        "select_overview": "اختر علامة وفئة لرؤية نظره عامه على السوق.",
        "analyzer_title": "🔍 محلل الإعلانات",
        "analyzer_desc": "الصق عنوان إعلان عربي وسيقوم التطبيق تلقائياً باستخراج العلامة والفئة والتخزين والحالة — ثم التنبؤ بالسعر.",
        "analyzer_input_label": "الصق عنوان الإعلان (عربي)",
        "analyzer_input_placeholder": "مثال: ايفون 15 برو ماكس 256 مستعمل ممتاز بدل",
        "analyzer_analyze_btn": "حلل الإعلان",
        "analyzer_extracted": "المعلومات المستخرجة",
        "analyzer_brand": "العلامة",
        "analyzer_series": "الفئة",
        "analyzer_storage": "التخزين",
        "analyzer_condition": "الحالة",
        "analyzer_predicted": "السعر المتنبأ",
        "analyzer_range": "نطاق السعر",
        "analyzer_deal": "معدل الصفقة",
        "analyzer_no_title": "الرجاء إدخال عنوان إعلان للتحليل.",
        "similar_listings": "إعلانات مشابهة",
        "no_similar": "لم يتم العثور على إعلانات مشابهة في البيانات.",
        "confidence_interval": "فترة الثقة",
    },
}


def t(key, lang):
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)


st.set_page_config(
    page_title="Mobile Price Engine | Jordan",
    page_icon="📈",
    layout="wide",
)

# --- UI STYLING (DARK THEME) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    :root {
        --bg-color: #0D1117;
        --primary-color: #58A6FF;
        --card-bg: #161B22;
        --text-color: #C9D1D9;
        --subtle-text-color: #8B949E;
        --border-color: #30363D;
    }

    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        color: #F0F6FC !important;
    }

    section[data-testid="stSidebar"] {
        background-color: var(--card-bg);
        border-right: 1px solid var(--border-color);
    }

    .metric-card {
        background: var(--card-bg);
        padding: 1.5rem 2rem;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
    }

    .metric-card h3 {
        color: var(--subtle-text-color) !important;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .prediction-value {
        font-size: 3rem;
        font-weight: 700;
        color: #F0F6FC;
    }

    .currency {
        font-size: 1.5rem;
        color: var(--primary-color);
        margin-left: 0.5rem;
        font-weight: 600;
    }

    .stButton>button {
        background-image: linear-gradient(to right, #314755 0%, #26a0da  51%, #314755  100%);
        color: white !important;
        border-radius: 6px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        border: none !important;
        width: 100%;
        transition: 0.5s !important;
        background-size: 200% auto;
    }

    .stButton>button:hover {
        background-position: right center;
        color: #fff;
    }

    .confidence-bar {
        background: linear-gradient(to right, #1a3a1a, #3FB950, #D29922, #F85149);
        height: 8px;
        border-radius: 4px;
        margin: 0.5rem 0;
        position: relative;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {"mae": 57.39, "mae_std": 0, "r2": 0}


@st.cache_data
def load_data():
    if not CLEANED_DATA_PATH.exists():
        return None
    df = pd.read_csv(CLEANED_DATA_PATH)
    df["brand"] = df["brand"].fillna("Other").astype(str)
    df["series"] = df["series"].fillna("Other").astype(str)
    return df


def _predict(model, brand, series, storage, condition, phone_age_months=0,
             is_pro=False, is_max=False, is_ultra=False, is_plus=False,
             has_warranty=False, is_sealed=False):
    """Run a single prediction and return the price in JOD."""
    title_parts = [brand, series, f"{storage}GB", condition]
    if is_pro:
        title_parts.append("pro")
    if is_max:
        title_parts.append("max")
    if is_ultra:
        title_parts.append("ultra")
    if is_plus:
        title_parts.append("plus")
    if has_warranty:
        title_parts.append("كفالة")
    if is_sealed:
        title_parts.append("كرتون مسكر")

    input_df = pd.DataFrame([{
        "title": " ".join(title_parts),
        "brand": brand,
        "series": series,
        "storage_gb": storage,
        "phone_age_months": phone_age_months,
        "condition": condition,
        "model_name": f"{brand} {series}",
        "color_actual": "Unknown",
    }])

    prediction = np.expm1(model.predict(input_df)[0])
    if not np.isfinite(prediction) or prediction <= 0:
        return None
    return prediction


def _find_similar_listings(df, brand, series, condition, prediction, n=5):
    """Find the N closest real listings by brand + series + condition."""
    mask = (df["brand"] == brand) & (df["series"] == series)
    if condition in df["condition"].unique():
        mask = mask & (df["condition"] == condition)
    similar = df[mask].copy()
    if similar.empty:
        # Fallback: just brand + series
        similar = df[(df["brand"] == brand) & (df["series"] == series)].copy()
    if similar.empty or prediction is None:
        return similar.head(0)
    similar["price_diff"] = abs(similar["price_jd"] - prediction)
    similar = similar.sort_values("price_diff").head(n)
    return similar[["title", "price_jd", "condition", "storage_gb"]]


def main():
    # Language toggle — stored in session state for persistence
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"

    lang = st.sidebar.selectbox(
        "🌐 Language / اللغة",
        ["en", "ar"],
        index=0 if st.session_state["lang"] == "en" else 1,
        format_func=lambda x: "English" if x == "en" else "العربية",
        key="lang_selector",
    )
    st.session_state["lang"] = lang

    st.title(t("title", lang))
    st.markdown(f"<p style='color: #8B949E;'>{t('subtitle', lang)}</p>", unsafe_allow_html=True)

    df = load_data()
    model = load_model()
    metrics = load_metrics()

    if df is None:
        st.error(f"Data not found at {CLEANED_DATA_PATH}. Please run `src/cleaning.py` first.")
        return
    if model is None:
        st.error(f"Model not found at {MODEL_PATH}. Please run `src/train_model.py` first.")
        return

    # --- SIDEBAR ---
    with st.sidebar:
        st.header(t("device_specs", lang))

        brand = st.selectbox(t("brand", lang), sorted(df["brand"].unique()))

        brand_df = df[df["brand"] == brand]
        series_counts = brand_df["series"].value_counts()
        available_series = sorted(series_counts[series_counts >= 5].index)
        series = st.selectbox(t("series", lang), available_series)

        storage = st.select_slider(
            t("storage", lang),
            options=["All", 32, 64, 128, 256, 512, 1024],
            value="All",
        )

        condition = st.selectbox(
            t("condition", lang),
            ["جديد", "مستعمل - ممتاز", "مستعمل - جيد", "مستعمل - سيئ"],
        )

        st.markdown("---")
        st.subheader(t("asking_price", lang))
        asking_price = st.number_input(
            t("seller_price", lang),
            min_value=0, max_value=2000, value=0, step=10,
            help="Enter the seller's asking price to get a Deal Score",
        )

        st.subheader(t("premium_features", lang))
        is_pro = st.checkbox("Pro / برو") if "apple" in brand.lower() or "huawei" in brand.lower() else False
        is_max = st.checkbox("Max / ماكس") if "apple" in brand.lower() else False
        is_ultra = st.checkbox("Ultra / الترا") if "samsung" in brand.lower() or "xiaomi" in brand.lower() else False
        is_plus = st.checkbox("Plus / بلس")

        st.subheader(t("device_attributes", lang))
        has_warranty = st.checkbox(t("warranty", lang))
        is_sealed = st.checkbox(t("sealed", lang))

        st.markdown("---")
        predict_btn = st.button(t("calculate", lang))

    # --- MAIN CONTENT ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 Price Estimate", "📊 Market Analysis",
        "🔄 Trade-In Calculator", "⚖️ Brand Comparison",
        "🔍 " + t("analyzer_title", lang),
    ])

    # ═══════════════════════════════════════════════════════════════
    # TAB 1: Price Estimate
    # ═══════════════════════════════════════════════════════════════
    with tab1:
        series_listings = df[(df["brand"] == brand) & (df["series"] == series)]

        def _show_market_overview():
            if series_listings.empty:
                st.info(t("no_listings", lang))
                return
            st.subheader(f"{brand} {series} — {t('market_overview', lang)}")
            col_avg, col_med, col_cnt = st.columns(3)
            with col_avg:
                st.markdown(f"""
                    <div class='metric-card'>
                        <h3>{t('avg_price', lang)}</h3>
                        <p class='prediction-value'>{series_listings['price_jd'].mean():,.0f}<span class='currency'>JOD</span></p>
                    </div>
                """, unsafe_allow_html=True)
            with col_med:
                st.markdown(f"""
                    <div class='metric-card'>
                        <h3>{t('median_price', lang)}</h3>
                        <p class='prediction-value'>{series_listings['price_jd'].median():,.0f}<span class='currency'>JOD</span></p>
                    </div>
                """, unsafe_allow_html=True)
            with col_cnt:
                st.markdown(f"""
                    <div class='metric-card'>
                        <h3>{t('listings_count', lang)}</h3>
                        <p class='prediction-value'>{len(series_listings)}<span class='currency'></span></p>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("#### " + t("price_by_storage", lang))
            breakdown = series_listings.groupby(["storage_gb", "condition"]).agg(
                avg_price=("price_jd", "mean"),
                min_price=("price_jd", "min"),
                max_price=("price_jd", "max"),
                count=("price_jd", "count"),
            ).reset_index()
            breakdown = breakdown[breakdown["count"] >= 2].sort_values(
                ["storage_gb", "avg_price"], ascending=[True, False]
            )
            if not breakdown.empty:
                cols_renamed = {
                    "storage_gb": t("storage", lang),
                    "condition": t("condition", lang),
                    "avg_price": f"{t('avg_price', lang).split()[0]} (JOD)",
                    "min_price": "Min (JOD)",
                    "max_price": "Max (JOD)",
                    "count": t("listings_count", lang),
                }
                st.dataframe(
                    breakdown.rename(columns=cols_renamed).round(0),
                    width="stretch", hide_index=True,
                )
            st.markdown("#### " + t("all_listings", lang))
            st.dataframe(
                series_listings[["title", "price_jd", "condition", "storage_gb"]].sort_values("price_jd"),
                width="stretch", hide_index=True,
            )

        if storage == "All":
            _show_market_overview()
        elif predict_btn:
            with st.spinner("Predicting..."):
                try:
                    prediction = _predict(
                        model, brand, series, storage, condition,
                        is_pro=is_pro, is_max=is_max, is_ultra=is_ultra,
                        is_plus=is_plus, has_warranty=has_warranty, is_sealed=is_sealed,
                    )
                except Exception as e:
                    st.error(f"Prediction failed: {e}. Please check that the model is compatible with your inputs.")
                    return

            if prediction is None:
                st.warning("The model returned an invalid prediction for these inputs. Try different specifications.")
                return

            mae = metrics["mae"]

            col_pred, col_range, col_score = st.columns(3)

            with col_pred:
                st.markdown(f"""
                    <div class='metric-card'>
                        <h3>Predicted Market Value</h3>
                        <p class='prediction-value'>{prediction:,.0f}<span class='currency'>JOD</span></p>
                    </div>
                """, unsafe_allow_html=True)

            with col_range:
                lower = prediction - mae
                upper = prediction + mae
                # Confidence interval bar visualization
                st.markdown(f"""
                    <div class='metric-card'>
                        <h3>{t('confidence_interval', lang)}</h3>
                        <p style='font-size: 1.1rem; color: #8B949E;'>{lower:,.0f} JOD</p>
                        <div style='background: linear-gradient(to right, #58A6FF, #F0F6FC, #58A6FF);
                                    height: 6px; border-radius: 3px; margin: 0.3rem 0; position: relative;'>
                            <div style='position: absolute; left: 50%; top: -4px;
                                        width: 2px; height: 14px; background: #F0F6FC;'></div>
                        </div>
                        <p style='font-size: 1.8rem; font-weight: 700; text-align: center;'>
                            {prediction:,.0f} <span style='font-size: 1rem; color: #58A6FF;'>JOD</span>
                        </p>
                        <p style='font-size: 1.1rem; color: #8B949E; text-align: right;'>{upper:,.0f} JOD</p>
                        <p style='font-size: 0.85rem; color: #8B949E; text-align: center; margin-top: 0.3rem;'>
                            ±{mae:,.0f} JOD (model MAE)
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            with col_score:
                if asking_price > 0:
                    deal_score = (prediction - asking_price) / prediction * 100
                    if deal_score > 10:
                        label = "🔥 Great Deal"
                        color = "#3FB950"
                    elif deal_score > 0:
                        label = "👍 Fair Price"
                        color = "#D29922"
                    elif deal_score > -10:
                        label = "⚠️ Slightly Overpriced"
                        color = "#F85149"
                    else:
                        label = "🚨 Overpriced"
                        color = "#F85149"
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h3>Deal Score</h3>
                            <p style='font-size: 1.8rem; font-weight: 700; color: {color};'>{deal_score:+.0f}%</p>
                            <p style='color: {color}; font-weight: 600;'>{label}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class='metric-card'>
                            <h3>Deal Score</h3>
                            <p style='font-size: 1.2rem; color: #8B949E;'>Enter an asking price to evaluate</p>
                        </div>
                    """, unsafe_allow_html=True)

            # Similar Listings
            st.subheader(t("similar_listings", lang))
            similar = _find_similar_listings(df, brand, series, condition, prediction)
            if not similar.empty:
                st.dataframe(similar, width="stretch")
            else:
                st.info(t("no_similar", lang))
        else:
            st.info(t("adjust_sidebar", lang))

    # ═══════════════════════════════════════════════════════════════
    # TAB 2: Market Analysis
    # ═══════════════════════════════════════════════════════════════
    with tab2:
        st.subheader(f"Market Trends: {brand}")
        brand_df = df[df["brand"] == brand]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### " + t("listings_by_series", lang))
            series_counts_plot = brand_df["series"].value_counts().reset_index()
            fig_series = px.bar(series_counts_plot, x="series", y="count", color="series", template="plotly_dark")
            st.plotly_chart(fig_series, width="stretch")
        with col2:
            st.markdown("#### " + t("price_dist", lang))
            fig_dist = px.histogram(brand_df, x="price_jd", nbins=30, template="plotly_dark", color_discrete_sequence=["#58A6FF"])
            st.plotly_chart(fig_dist, width="stretch")

        st.markdown("#### " + t("price_by_condition", lang))
        fig_box = px.box(brand_df, x="condition", y="price_jd", color="condition", template="plotly_dark")
        st.plotly_chart(fig_box, width="stretch")

    # ═══════════════════════════════════════════════════════════════
    # TAB 3: Trade-In Calculator
    # ═══════════════════════════════════════════════════════════════
    with tab3:
        st.subheader(t("trade_in_title", lang))
        st.markdown(t("trade_in_desc", lang))

        has_current_phone = st.checkbox(
            "I have a phone to trade in" if lang == "en" else "عندي هاتف للمقايضة",
            value=True, key="has_current_phone",
        )

        cur_value = 0.0
        cur_brand = None
        cur_series = None

        if has_current_phone:
            col_current, col_budget = st.columns(2)
            with col_current:
                st.markdown("#### " + ("Your Current Phone" if lang == "en" else "هاتفك الحالي"))
                cur_brand = st.selectbox(
                    "Current Brand" if lang == "en" else "العلامة الحالية",
                    sorted(df["brand"].unique()), key="cur_brand",
                )
                cur_brand_df = df[df["brand"] == cur_brand]
                cur_series_counts = cur_brand_df["series"].value_counts()
                cur_available = sorted(cur_series_counts[cur_series_counts >= 5].index)
                cur_series = st.selectbox(
                    "Current Series" if lang == "en" else "الفئة الحالية",
                    cur_available, key="cur_series",
                )
                cur_storage = st.select_slider(
                    "Current Storage" if lang == "en" else "سعة التخزين الحالية",
                    options=[32, 64, 128, 256, 512, 1024], value=128, key="cur_storage",
                )
                cur_condition = st.selectbox(
                    "Current Condition" if lang == "en" else "الحالة الحالية",
                    ["مستعمل - ممتاز", "مستعمل - جيد", "مستعمل - سيئ"], key="cur_cond",
                )
            with col_budget:
                st.markdown("#### " + t("upgrade_budget", lang))
                extra_budget = st.number_input(
                    t("extra_budget", lang),
                    min_value=0, max_value=3000, value=300, step=50,
                )
        else:
            extra_budget = st.number_input(
                t("extra_budget", lang),
                min_value=0, max_value=3000, value=500, step=50,
            )

        calc_label = t("calc_trade", lang)
        if st.button(calc_label, key="trade_in_btn"):
            with st.spinner("Calculating..."):
                if has_current_phone:
                    try:
                        cur_value = _predict(model, cur_brand, cur_series, cur_storage, cur_condition)
                    except Exception:
                        cur_value = None

                    if cur_value is None:
                        st.error(
                            "Cannot estimate your current phone's value. Try different specs."
                            if lang == "en"
                            else "لا يمكن تقدير قيمة هاتفك. جرب مواصفات مختلفة."
                        )
                        return

            total_budget = cur_value + extra_budget

            if has_current_phone:
                st.markdown(f"""
                    <div class='metric-card'>
                        <h3>{t("trade_in_value", lang)}</h3>
                        <p class='prediction-value'>{cur_value:,.0f}<span class='currency'>JOD</span></p>
                        <p style='color: #8B949E;'>+ {extra_budget:,.0f} JOD = <strong>{total_budget:,.0f} JOD total</strong></p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='metric-card'>
                        <h3>{t("upgrade_budget", lang)}</h3>
                        <p class='prediction-value'>{total_budget:,.0f}<span class='currency'>JOD</span></p>
                    </div>
                """, unsafe_allow_html=True)

            upgrade_pool = df[df["price_jd"] <= total_budget]
            if has_current_phone and cur_brand and cur_series:
                upgrade_pool = upgrade_pool[
                    ~((upgrade_pool["brand"] == cur_brand) & (upgrade_pool["series"] == cur_series))
                ]
                upgrade_pool = upgrade_pool[upgrade_pool["price_jd"] > cur_value * 0.8]

            if not upgrade_pool.empty:
                top_upgrades = upgrade_pool.groupby(["brand", "series"]).agg(
                    avg_price=("price_jd", "mean"),
                    count=("price_jd", "count"),
                ).reset_index()
                top_upgrades = top_upgrades[top_upgrades["count"] >= 3].sort_values("avg_price", ascending=False).head(10)
                st.markdown("#### " + t("upgrade_options", lang))
                st.dataframe(top_upgrades.rename(columns={
                    "brand": "Brand", "series": "Series",
                    "avg_price": "Avg Price (JOD)", "count": "Listings",
                }), width="stretch", hide_index=True)
            else:
                st.info(t("no_upgrades", lang))

    # ═══════════════════════════════════════════════════════════════
    # TAB 4: Brand Comparison
    # ═══════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("⚖️ Brand Value Comparison")
        st.markdown("Compare how different brands hold their value in the Jordanian market.")

        col_a, col_b = st.columns(2)
        with col_a:
            brand_a = st.selectbox("Brand A", sorted(df["brand"].unique()), key="brand_a")
            df_a = df[df["brand"] == brand_a]
        with col_b:
            brand_b = st.selectbox("Brand B", sorted(df["brand"].unique()), index=1, key="brand_b")
            df_b = df[df["brand"] == brand_b]

        if st.button("Compare Brands", key="compare_btn"):
            with st.spinner("Comparing..."):
                col_stat_a, col_stat_b = st.columns(2)
                stats_a = {
                    "listings": len(df_a),
                    "avg_price": df_a["price_jd"].mean(),
                    "median_price": df_a["price_jd"].median(),
                    "pct_new": (df_a["condition"] == "جديد").mean() * 100 if "condition" in df_a.columns else 0,
                }
                stats_b = {
                    "listings": len(df_b),
                    "avg_price": df_b["price_jd"].mean(),
                    "median_price": df_b["price_jd"].median(),
                    "pct_new": (df_b["condition"] == "جديد").mean() * 100 if "condition" in df_b.columns else 0,
                }

            with col_stat_a:
                st.markdown(f"""
                    <div class='metric-card'>
                        <h3>{brand_a}</h3>
                        <p style='font-size: 1.5rem; font-weight: 700;'>{stats_a['avg_price']:,.0f} JOD avg</p>
                        <p style='color: #8B949E;'>{stats_a['listings']} listings | {stats_a['pct_new']:.0f}% new</p>
                    </div>
                """, unsafe_allow_html=True)
            with col_stat_b:
                st.markdown(f"""
                    <div class='metric-card'>
                        <h3>{brand_b}</h3>
                        <p style='font-size: 1.5rem; font-weight: 700;'>{stats_b['avg_price']:,.0f} JOD avg</p>
                        <p style='color: #8B949E;'>{stats_b['listings']} listings | {stats_b['pct_new']:.0f}% new</p>
                    </div>
                """, unsafe_allow_html=True)

            compare_df = pd.concat([
                df_a[["condition", "price_jd"]].assign(brand=brand_a),
                df_b[["condition", "price_jd"]].assign(brand=brand_b),
            ])
            fig_compare = px.box(
                compare_df, x="condition", y="price_jd", color="brand",
                template="plotly_dark", title="Price Distribution by Condition",
            )
            st.plotly_chart(fig_compare, width="stretch")

    # ═══════════════════════════════════════════════════════════════
    # TAB 5: My Listing Analyzer
    # ═══════════════════════════════════════════════════════════════
    with tab5:
        st.subheader(t("analyzer_title", lang))
        st.markdown(t("analyzer_desc", lang))

        raw_title = st.text_input(
            t("analyzer_input_label", lang),
            placeholder=t("analyzer_input_placeholder", lang),
            key="analyzer_input",
        )

        analyzer_asking = st.number_input(
            t("seller_price", lang),
            min_value=0, max_value=2000, value=0, step=10,
            key="analyzer_asking",
            help="Optional: enter asking price for deal score",
        )

        if st.button(t("analyzer_analyze_btn", lang), key="analyzer_btn"):
            if not raw_title.strip():
                st.warning(t("analyzer_no_title", lang))
            else:
                with st.spinner("Analyzing..."):
                    # Step 1: Extract brand from title
                    normalized = normalize_arabic(raw_title)
                    detected_brand = "Other"
                    brand_patterns = [
                        (r"ايفون|iphone|apple|ابل", "Apple"),
                        (r"سامسونج|samsung|جالاكسي", "Samsung"),
                        (r"شاومي|xiaomi|ريدمي|بوكو", "Xiaomi"),
                        (r"هواوي|huawei", "Huawei"),
                        (r"هونور|honor", "Honor"),
                        (r"اوبو|oppo", "OPPO"),
                        (r"تكنو|tecno", "Tecno"),
                        (r"انفينيكس|infinix", "Infinix"),
                        (r"نوكيا|nokia", "Nokia"),
                        (r"ريلمي|realme", "Realme"),
                        (r"فيفو|vivo", "Vivo"),
                        (r"جوجل|google|بيكسل|pixel", "Google"),
                        (r"ون\s*بلس|oneplus", "OnePlus"),
                    ]
                    for pattern, bname in brand_patterns:
                        if re.search(pattern, normalized):
                            detected_brand = bname
                            break

                    # Step 2: Extract series
                    detected_series = extract_series(raw_title, detected_brand)

                    # Step 3: Extract storage
                    detected_storage = extract_storage_gb("", raw_title)
                    if detected_storage is None:
                        detected_storage = 128  # default

                    # Step 4: Extract condition
                    detected_condition = normalize_condition(raw_title)
                    if detected_condition == "Unknown":
                        detected_condition = "مستعمل - ممتاز"  # default

                # Display extracted info
                st.markdown("### " + t("analyzer_extracted", lang))
                col_b, col_s, col_st, col_c = st.columns(4)
                with col_b:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h3>{t('analyzer_brand', lang)}</h3>
                            <p style='font-size: 1.5rem; font-weight: 700;'>{detected_brand}</p>
                        </div>
                    """, unsafe_allow_html=True)
                with col_s:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h3>{t('analyzer_series', lang)}</h3>
                            <p style='font-size: 1.2rem; font-weight: 700; font-size: 0.95rem;'>{detected_series}</p>
                        </div>
                    """, unsafe_allow_html=True)
                with col_st:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h3>{t('analyzer_storage', lang)}</h3>
                            <p style='font-size: 1.5rem; font-weight: 700;'>{detected_storage} GB</p>
                        </div>
                    """, unsafe_allow_html=True)
                with col_c:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <h3>{t('analyzer_condition', lang)}</h3>
                            <p style='font-size: 1.2rem; font-weight: 700;'>{detected_condition}</p>
                        </div>
                    """, unsafe_allow_html=True)

                # Predict
                with st.spinner("Predicting price..."):
                    try:
                        pred = _predict(model, detected_brand, detected_series, detected_storage, detected_condition)
                    except Exception as e:
                        pred = None
                        st.error(f"Prediction failed: {e}")

                if pred is not None:
                    mae = metrics["mae"]
                    st.markdown("### " + t("analyzer_predicted", lang))
                    col_p, col_r = st.columns(2)
                    with col_p:
                        st.markdown(f"""
                            <div class='metric-card'>
                                <h3>{t('analyzer_predicted', lang)}</h3>
                                <p class='prediction-value'>{pred:,.0f}<span class='currency'>JOD</span></p>
                            </div>
                        """, unsafe_allow_html=True)
                    with col_r:
                        st.markdown(f"""
                            <div class='metric-card'>
                                <h3>{t('analyzer_range', lang)}</h3>
                                <p style='font-size: 1.8rem; font-weight: 700;'>
                                    {pred - mae:,.0f} – {pred + mae:,.0f}
                                    <span style='font-size: 1rem; color: #58A6FF;'>JOD</span>
                                </p>
                            </div>
                        """, unsafe_allow_html=True)

                    # Deal score
                    if analyzer_asking > 0:
                        deal_score = (pred - analyzer_asking) / pred * 100
                        if deal_score > 10:
                            label = "🔥 Great Deal"
                            color = "#3FB950"
                        elif deal_score > 0:
                            label = "👍 Fair Price"
                            color = "#D29922"
                        elif deal_score > -10:
                            label = "⚠️ Slightly Overpriced"
                            color = "#F85149"
                        else:
                            label = "🚨 Overpriced"
                            color = "#F85149"
                        st.markdown(f"""
                            <div class='metric-card'>
                                <h3>{t('analyzer_deal', lang)}</h3>
                                <p style='font-size: 1.8rem; font-weight: 700; color: {color};'>{deal_score:+.0f}%</p>
                                <p style='color: {color}; font-weight: 600;'>{label}</p>
                            </div>
                        """, unsafe_allow_html=True)

                    # Similar listings
                    st.markdown("#### " + t("similar_listings", lang))
                    similar = _find_similar_listings(df, detected_brand, detected_series, detected_condition, pred)
                    if not similar.empty:
                        st.dataframe(similar, width="stretch")
                    else:
                        st.info(t("no_similar", lang))


if __name__ == "__main__":
    main()
