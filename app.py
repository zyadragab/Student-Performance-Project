import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─────────────────────────────────────────────
st.set_page_config(page_title="Student Performance", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main-title { font-family: 'DM Serif Display', serif; font-size: 2.2rem; color: #1a1a2e; margin-bottom: 0; }
    .sub-title  { font-size: 1rem; color: #666; margin-top: 4px; margin-bottom: 1.5rem; }
    div[data-testid="stSidebar"] { background: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def grade_label(g):
    if g >= 17: return "Excellent", "#155724", "#d4edda"
    if g >= 14: return "Good",      "#004085", "#cce5ff"
    if g >= 10: return "Pass",      "#856404", "#fff3cd"
    return              "Fail",      "#721c24", "#f8d7da"

@st.cache_resource
def train_models(df_raw):
    df = df_raw.copy()

    # Preprocessing
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    # Outlier treatment — absences only
    Q1 = df['absences'].quantile(0.25)
    Q3 = df['absences'].quantile(0.75)
    df['absences'] = df['absences'].clip(upper=Q3 + 1.5*(Q3-Q1))

    # Feature selection
    X = df.drop(columns=['G3','school','reason','guardian','nursery','Pstatus','sex','age'])
    y = df['G3']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Train all 5 models — exactly as in notebook
    en    = ElasticNet(alpha=0.5, l1_ratio=0.999)
    ridge = Ridge(alpha=30)
    lasso = Lasso(alpha=0.5)
    svr   = SVR(kernel='linear', C=500, epsilon=1)
    rf    = RandomForestRegressor(n_estimators=200, max_depth=4,
                                   min_samples_leaf=10, max_features=0.7,
                                   max_samples=0.8, random_state=42)

    en.fit(X_train_sc, y_train);    en_pred    = en.predict(X_test_sc)
    ridge.fit(X_train_sc, y_train); ridge_pred = ridge.predict(X_test_sc)
    lasso.fit(X_train_sc, y_train); lasso_pred = lasso.predict(X_test_sc)
    svr.fit(X_train_sc, y_train);   svr_pred   = svr.predict(X_test_sc)
    rf.fit(X_train, y_train);       rf_pred    = rf.predict(X_test)

    models_info = {
        'ElasticNet':    {'model': en,    'scaled': True,  'pred': en_pred,    'color': '#4C72B0'},
        'Ridge':         {'model': ridge, 'scaled': True,  'pred': ridge_pred, 'color': '#5C8DD6'},
        'Lasso':         {'model': lasso, 'scaled': True,  'pred': lasso_pred, 'color': '#6B4FA0'},
        'SVR':           {'model': svr,   'scaled': True,  'pred': svr_pred,   'color': '#55A868'},
        'Random Forest': {'model': rf,    'scaled': False, 'pred': rf_pred,    'color': '#DD8452'},
    }

    results = {}
    for name, info in models_info.items():
        pred = info['pred']
        results[name] = {
            **info,
            'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
            'MAE':  mean_absolute_error(y_test, pred),
            'R²':   r2_score(y_test, pred),
        }

    return results, scaler, X, X_train, X_test, y_train, y_test

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 Student Performance")
    st.markdown("---")
    uploaded = st.file_uploader("Upload student_data.csv", type="csv")
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.success(f"✅ {len(df_raw)} records loaded")
    else:
        df_raw = pd.read_csv("student_data.csv")
        st.info(f"Using default dataset ({len(df_raw)} students)")

    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview",
        "📊 Model Comparison",
        "🔍 Overfitting Check",
        "🔮 Predict Grade",
        "📈 Feature Importance",
        "🏆 Best Model Results",
    ])

# ─────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────
with st.spinner("Training models..."):
    results, scaler, X, X_train, X_test, y_train, y_test = train_models(df_raw)

comparison = pd.DataFrame({
    name: {'RMSE': v['RMSE'], 'MAE': v['MAE'], 'R²': v['R²']}
    for name, v in results.items()
}).T.round(3)

best_name = comparison['RMSE'].idxmin()

# ═══════════════════════════════════════════════
# PAGE: Overview
# ═══════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<p class="main-title">🎓 Student Grade Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">5 ML models trained to predict final grade G3</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students",     len(df_raw))
    c2.metric("Features",     X.shape[1])
    c3.metric("Test set",     len(y_test))
    c4.metric("Best model",   best_name)

    st.markdown("---")

    # G3 Distribution + Balance
    df2 = df_raw.copy()
    le  = LabelEncoder()
    for col in df2.select_dtypes(include='object').columns:
        df2[col] = le.fit_transform(df2[col])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("G3 Distribution")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df2['G3'], bins=20, color='#4C72B0', edgecolor='white', linewidth=0.8)
        ax.axvline(df2['G3'].mean(),   color='red',    linestyle='--', label=f"Mean: {df2['G3'].mean():.1f}")
        ax.axvline(df2['G3'].median(), color='orange', linestyle='--', label=f"Median: {df2['G3'].median():.1f}")
        ax.set_xlabel('Final Grade (G3)')
        ax.legend(); ax.spines[['top','right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("G3 Class Balance")
        bins_   = [0, 9, 13, 16, 20]
        labels_ = ['Fail\n(0-9)', 'Pass\n(10-13)', 'Good\n(14-16)', 'Excellent\n(17-20)']
        cats    = pd.cut(df2['G3'], bins=bins_, labels=labels_)
        counts  = cats.value_counts().reindex(labels_)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors_cat = ['#C44E52','#DD8452','#4C72B0','#55A868']
        bars = ax.bar(labels_, counts.values, color=colors_cat, edgecolor='white', linewidth=1.2)
        for bar, v in zip(bars, counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    str(v), ha='center', fontweight='bold', fontsize=10)
        ax.set_ylabel('Count'); ax.spines[['top','right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Models")
    descs = {
        'ElasticNet':    "Linear model combining L1 + L2 regularization. Handles correlated features well.",
        'Ridge':         "Linear model with L2 regularization. Shrinks all coefficients without elimination.",
        'Lasso':         "Linear model with L1 regularization. Automatically zeroes out irrelevant features.",
        'SVR':           "Support Vector Regressor with linear kernel. Works well on scaled data.",
        'Random Forest': "Ensemble of 200 decision trees. Handles non-linearity and feature interactions.",
    }
    c1, c2 = st.columns(2)
    for i, (name, desc) in enumerate(descs.items()):
        col = c1 if i % 2 == 0 else c2
        with col:
            with st.container(border=True):
                if name == best_name:
                    st.success(f"⭐ {name} — Best Model")
                else:
                    st.markdown(f"**{name}**")
                st.caption(desc)
                st.markdown(f"RMSE: `{comparison.loc[name,'RMSE']:.3f}` | R²: `{comparison.loc[name,'R²']:.3f}`")

# ═══════════════════════════════════════════════
# PAGE: Model Comparison
# ═══════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.markdown('<p class="main-title">📊 Model Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Side-by-side metrics for all 5 models</p>', unsafe_allow_html=True)

    st.dataframe(
        comparison.style
            .highlight_min(subset=['RMSE','MAE'], color='#d4edda')
            .highlight_max(subset=['R²'],          color='#d4edda'),
        use_container_width=True
    )

    st.markdown("---")
    names  = list(results.keys())
    colors = [results[n]['color'] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ['RMSE','MAE','R²']):
        vals = [results[n][metric] for n in names]
        bars = ax.bar(names, vals, color=colors, edgecolor='white', linewidth=1.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.02,
                    f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(vals)*1.2)
        ax.tick_params(axis='x', rotation=20)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Model Comparison — All 5 Models', fontsize=14, fontweight='bold')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Actual vs Predicted — per model")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, name in zip(axes, names):
        pred = results[name]['pred']
        ax.scatter(y_test, pred, alpha=0.5, color=results[name]['color'], edgecolors='white', s=30)
        mn, mx = y_test.min(), y_test.max()
        ax.plot([mn,mx],[mn,mx],'k--',lw=1.5)
        ax.set_title(name, fontweight='bold', fontsize=9)
        ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
        ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════
# PAGE: Overfitting Check
# ═══════════════════════════════════════════════
elif page == "🔍 Overfitting Check":
    st.markdown('<p class="main-title">🔍 Overfitting Check</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Train RMSE vs Test RMSE for all models</p>', unsafe_allow_html=True)

    data = []
    for name, info in results.items():
        model  = info['model']
        scaled = info['scaled']
        Xtr = scaler.transform(X_train) if scaled else X_train
        Xte = scaler.transform(X_test)  if scaled else X_test
        tr  = np.sqrt(mean_squared_error(y_train, model.predict(Xtr)))
        te  = np.sqrt(mean_squared_error(y_test,  model.predict(Xte)))
        diff = te - tr
        status = '✅ OK' if diff < 0.5 else '⚠️ Overfit'
        data.append({'Model': name, 'Train RMSE': round(tr,3),
                     'Test RMSE': round(te,3), 'Difference': round(diff,3), 'Status': status})

    df_ov = pd.DataFrame(data)
    st.dataframe(df_ov, use_container_width=True, hide_index=True)

    st.markdown("---")
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(df_ov))
    w = 0.35
    ax.bar(x-w/2, df_ov['Train RMSE'], w, label='Train RMSE', color='#4C72B0', edgecolor='white')
    ax.bar(x+w/2, df_ov['Test RMSE'],  w, label='Test RMSE',  color='#C44E52', edgecolor='white', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(df_ov['Model'], rotation=15)
    ax.set_ylabel('RMSE')
    ax.set_title('Train vs Test RMSE', fontweight='bold')
    ax.legend(); ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════
# PAGE: Predict Grade
# ═══════════════════════════════════════════════
elif page == "🔮 Predict Grade":
    st.markdown('<p class="main-title">🔮 Predict a Student\'s Grade</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Fill in the details and get predictions from all 5 models</p>', unsafe_allow_html=True)

    with st.form("pred_form"):
        st.subheader("📚 Grades")
        c1, c2 = st.columns(2)
        g1 = c1.slider("G1 — First period",  0, 20, 12)
        g2 = c2.slider("G2 — Second period", 0, 20, 12)

        st.subheader("👤 Personal")
        c1, c2, c3 = st.columns(3)
        studytime  = c1.slider("Study time (1–4)",      1, 4,  2)
        failures   = c2.slider("Past failures",          0, 4,  0)
        absences   = c3.slider("Absences",               0, 19, 4)
        famrel     = c1.slider("Family rel (1–5)",       1, 5,  4)
        freetime   = c2.slider("Free time (1–5)",        1, 5,  3)
        goout      = c3.slider("Go out (1–5)",           1, 5,  3)
        dalc       = c1.slider("Weekday alcohol (1–5)",  1, 5,  1)
        walc       = c2.slider("Weekend alcohol (1–5)",  1, 5,  1)
        health     = c3.slider("Health (1–5)",           1, 5,  3)
        traveltime = c1.slider("Travel time (1–4)",      1, 4,  1)

        st.subheader("🏠 Family")
        c1, c2, c3 = st.columns(3)
        medu    = c1.slider("Mother edu (0–4)", 0, 4, 2)
        fedu    = c2.slider("Father edu (0–4)", 0, 4, 2)
        mjob    = c3.selectbox("Mother job", [0,1,2,3,4],
                               format_func=lambda x: ['at_home','health','other','services','teacher'][x])
        fjob    = c1.selectbox("Father job", [0,1,2,3,4],
                               format_func=lambda x: ['at_home','health','other','services','teacher'][x])
        famsize = c2.selectbox("Family size", [0,1], format_func=lambda x: ['LE3','GT3'][x])

        st.subheader("✅ Yes / No")
        c1, c2, c3, c4 = st.columns(4)
        address    = c1.selectbox("Address",        [0,1], format_func=lambda x: ['Rural','Urban'][x])
        schoolsup  = c2.selectbox("School support", [0,1], format_func=lambda x: ['No','Yes'][x])
        famsup     = c3.selectbox("Family support", [0,1], format_func=lambda x: ['No','Yes'][x])
        paid       = c4.selectbox("Paid classes",   [0,1], format_func=lambda x: ['No','Yes'][x])
        activities = c1.selectbox("Activities",     [0,1], format_func=lambda x: ['No','Yes'][x])
        higher     = c2.selectbox("Higher edu",     [0,1], format_func=lambda x: ['No','Yes'][x])
        internet   = c3.selectbox("Internet",       [0,1], format_func=lambda x: ['No','Yes'][x])
        romantic   = c4.selectbox("Romantic",       [0,1], format_func=lambda x: ['No','Yes'][x])

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        input_dict = {
            'address': address, 'famsize': famsize,
            'Medu': medu, 'Fedu': fedu, 'Mjob': mjob, 'Fjob': fjob,
            'traveltime': traveltime, 'studytime': studytime, 'failures': failures,
            'schoolsup': schoolsup, 'famsup': famsup, 'paid': paid,
            'activities': activities, 'higher': higher, 'internet': internet,
            'romantic': romantic, 'famrel': famrel, 'freetime': freetime,
            'goout': goout, 'Dalc': dalc, 'Walc': walc,
            'health': health, 'absences': absences,
            'G1': g1, 'G2': g2,
        }

        input_df = pd.DataFrame([input_dict])[X.columns]
        input_sc = scaler.transform(input_df)

        preds = {}
        for name, info in results.items():
            inp = input_sc if info['scaled'] else input_df
            preds[name] = float(np.clip(info['model'].predict(inp)[0], 0, 20))

        st.markdown("---")
        st.subheader("Predictions")
        cols = st.columns(5)
        for col, (name, pred) in zip(cols, preds.items()):
            label, tc, bc = grade_label(pred)
            with col:
                with st.container(border=True):
                    st.caption(name)
                    st.metric("Grade", f"{pred:.1f} / 20")
                    if label == "Excellent": st.success(label)
                    elif label == "Good":    st.info(label)
                    elif label == "Pass":    st.warning(label)
                    else:                    st.error(label)

        avg = np.mean(list(preds.values()))
        st.info(f"📊 Average across all models: **{avg:.1f} / 20**")

        fig, ax = plt.subplots(figsize=(9, 3))
        nms  = list(preds.keys())
        vls  = list(preds.values())
        clrs = [results[n]['color'] for n in nms]
        bars = ax.barh(nms, vls, color=clrs, edgecolor='white', height=0.5)
        ax.axvline(10, color='gray', linestyle='--', lw=1, alpha=0.5, label='Pass (10)')
        for bar, v in zip(bars, vls):
            ax.text(v+0.2, bar.get_y()+bar.get_height()/2,
                    f'{v:.1f}', va='center', fontweight='bold')
        ax.set_xlim(0, 22); ax.legend()
        ax.set_title('Predictions — All Models', fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════════
# PAGE: Feature Importance
# ═══════════════════════════════════════════════
elif page == "📈 Feature Importance":
    st.markdown('<p class="main-title">📈 Feature Importance</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Random Forest — which features matter most?</p>', unsafe_allow_html=True)

    rf_model = results['Random Forest']['model']
    imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    imp.head(10)[::-1].plot(kind='barh', color='#DD8452', ax=axes[0])
    axes[0].set_title('Top 10 Features', fontweight='bold')
    axes[0].set_xlabel('Importance')
    axes[0].spines[['top','right']].set_visible(False)

    cmap = plt.cm.RdYlGn
    norm = (imp - imp.min()) / (imp.max() - imp.min() + 1e-9)
    clrs = [cmap(v) for v in norm[::-1]]
    axes[1].barh(imp.index[::-1], imp.values[::-1], color=clrs, edgecolor='white')
    axes[1].set_title('All Features', fontweight='bold')
    axes[1].set_xlabel('Importance')
    axes[1].tick_params(axis='y', labelsize=8)
    axes[1].spines[['top','right']].set_visible(False)

    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.caption(f"Top 3: {', '.join(imp.head(3).index.tolist())}")

# ═══════════════════════════════════════════════
# PAGE: Best Model Results
# ═══════════════════════════════════════════════
elif page == "🏆 Best Model Results":
    st.markdown(f'<p class="main-title">🏆 Best Model — {best_name}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-title">RMSE: {comparison.loc[best_name,"RMSE"]:.3f} | MAE: {comparison.loc[best_name,"MAE"]:.3f} | R²: {comparison.loc[best_name,"R²"]:.3f}</p>', unsafe_allow_html=True)

    best_pred = results[best_name]['pred']
    actual    = y_test.values
    residuals = best_pred - actual

    # Plot 1: Scatter coloured by error
    st.subheader("Actual vs Predicted — coloured by error")
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(actual, best_pred, c=np.abs(residuals),
                    cmap='RdYlGn_r', edgecolors='white', s=60, alpha=0.85)
    plt.colorbar(sc, ax=ax, label='Absolute Error')
    ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', lw=1.5, label='Perfect')
    ax.set_xlabel('Actual G3'); ax.set_ylabel('Predicted G3')
    ax.set_title(f'{best_name} — Actual vs Predicted', fontweight='bold')
    ax.legend(); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Plot 2: Bar chart per student
    st.subheader("Actual vs Predicted — per student")
    x = np.arange(len(actual))
    w = 0.4
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.bar(x-w/2, actual,    w, label='Actual G3',    color='#4C72B0', edgecolor='white')
    ax.bar(x+w/2, best_pred, w, label='Predicted G3', color='#DD8452', edgecolor='white', alpha=0.85)
    ax.set_xlabel('Student Index'); ax.set_ylabel('G3 Grade')
    ax.set_title(f'{best_name} — Per Student', fontweight='bold')
    ax.legend(); ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Plot 3: Residual Plot
    st.subheader("Residual Plot")
    fig, ax = plt.subplots(figsize=(8, 4))
    clrs = ['#55A868' if abs(r) <= 1 else '#C44E52' for r in residuals]
    ax.scatter(best_pred, residuals, c=clrs, edgecolors='white', s=60, alpha=0.85)
    ax.axhline(0,  color='black', linestyle='--', lw=1.5)
    ax.axhline(1,  color='gray',  linestyle=':',  lw=1, alpha=0.6)
    ax.axhline(-1, color='gray',  linestyle=':',  lw=1, alpha=0.6)
    ax.set_xlabel('Predicted G3'); ax.set_ylabel('Residual (Predicted - Actual)')
    ax.set_title('Residual Plot', fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='#55A868', label='Error ≤ 1'),
                       Patch(facecolor='#C44E52', label='Error > 1')])
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Table
    st.subheader("Actual vs Predicted — table")
    df_res = pd.DataFrame({
        'Actual G3':    actual,
        'Predicted G3': best_pred.round(1),
        'Difference':   residuals.round(1)
    })
    df_res.index = range(1, len(df_res)+1)
    st.dataframe(df_res, use_container_width=True)
