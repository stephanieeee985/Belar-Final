import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)

st.set_page_config(page_title="Facility Reservation Analysis", page_icon="🏛️", layout="wide")
sns.set_theme(style="whitegrid")

# ----------------------------------------------------------------------------
# Data generation / loading
# ----------------------------------------------------------------------------
@st.cache_data
def generate_data(n_samples: int, seed: int) -> pd.DataFrame:
    np.random.seed(seed)
    data = {
        'Reservation_ID': [f"RES-{1000+i}" for i in range(n_samples)],
        'User_Category': np.random.choice(['Student', 'Faculty', 'Staff', 'Student Org', 'student'],
                                          size=n_samples, p=[0.4, 0.25, 0.15, 0.15, 0.05]),
        'Facility_Type': np.random.choice(['Auditorium', 'Lab', 'Conference Room', 'Lecture Hall'], size=n_samples),
        'Lead_Time_Days': np.random.choice([0, 1, 2, 5, 10, 20, -5, 999], size=n_samples,
                                           p=[0.15, 0.15, 0.1, 0.25, 0.2, 0.12, 0.01, 0.02]),
        'Participants': np.random.choice([10, 20, 50, 100, 0, -10], size=n_samples,
                                         p=[0.3, 0.3, 0.2, 0.15, 0.03, 0.02]),
        'Actual_Attendance': np.random.choice([10, 18, 45, 0, np.nan], size=n_samples,
                                              p=[0.3, 0.3, 0.2, 0.15, 0.05]),
        'Usage_Status': np.random.choice(['Used', 'Cancelled', 'No-Show'], size=n_samples, p=[0.6, 0.25, 0.15]),
        'Cancellation_Days_Before': np.random.choice([0, 1, 3, 5, np.nan], size=n_samples,
                                                     p=[0.1, 0.1, 0.1, 0.1, 0.6]),
    }
    df = pd.DataFrame(data)
    # inject 5 duplicate rows
    return pd.concat([df, df.iloc[:5]], ignore_index=True)


def categorize_lead_time(days):
    if days <= 2:
        return 'Last Minute'
    elif days <= 7:
        return 'Short Lead'
    elif days <= 14:
        return 'Medium Lead'
    return 'Early Booking'


@st.cache_data
def clean_data(df_raw: pd.DataFrame):
    df = df_raw.copy(deep=True)
    log = {}
    log['missing_before'] = df.isnull().sum()

    dup_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    log['dup_count'] = dup_count

    log['cat_before'] = df['User_Category'].value_counts(dropna=False)
    df['User_Category'] = df['User_Category'].replace(
        {'student': 'Student', 'STUDENT': 'Student', 'student org': 'Student Org'})
    log['cat_after'] = df['User_Category'].value_counts(dropna=False)

    invalid_lead = (df['Lead_Time_Days'] < 0) | (df['Lead_Time_Days'] > 180)
    invalid_part = df['Participants'] <= 0
    log['invalid_lead'] = int(invalid_lead.sum())
    log['invalid_part'] = int(invalid_part.sum())

    df.loc[invalid_lead, 'Lead_Time_Days'] = np.nan
    df.loc[invalid_part, 'Participants'] = np.nan
    df['Lead_Time_Days'] = df['Lead_Time_Days'].fillna(df['Lead_Time_Days'].median())
    df['Participants'] = df['Participants'].fillna(df['Participants'].median())
    log['missing_after'] = df.isnull().sum()

    # Feature engineering
    df['Lead_Time_Category'] = df['Lead_Time_Days'].apply(categorize_lead_time)
    df['Reservation_Outcome'] = df['Usage_Status'].apply(lambda x: 'Completed' if x == 'Used' else 'Not Completed')
    df['Attendance_Ratio'] = np.where(
        df['Participants'].isna() | (df['Participants'] == 0) | df['Actual_Attendance'].isna(),
        0.0, df['Actual_Attendance'] / df['Participants'])
    return df, log


@st.cache_data
def train_model(df: pd.DataFrame, max_depth: int, test_size: float, seed: int):
    X = df[['User_Category', 'Facility_Type', 'Lead_Time_Days', 'Participants', 'Lead_Time_Category']]
    y = df['Reservation_Outcome']
    X_enc = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=test_size, random_state=seed, stratify=y)
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision (Not Completed)': precision_score(y_test, y_pred, pos_label='Not Completed', zero_division=0),
        'Recall (Not Completed)': recall_score(y_test, y_pred, pos_label='Not Completed', zero_division=0),
        'F1 (Not Completed)': f1_score(y_test, y_pred, pos_label='Not Completed', zero_division=0),
    }
    cm = confusion_matrix(y_test, y_pred, labels=['Completed', 'Not Completed'])
    cm_df = pd.DataFrame(cm, index=['Actual Completed', 'Actual Not Completed'],
                         columns=['Pred Completed', 'Pred Not Completed'])
    report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).T
    importances = pd.Series(clf.feature_importances_, index=X_enc.columns).sort_values(ascending=False)
    return clf, list(X_enc.columns), metrics, cm_df, report, importances


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")
source = st.sidebar.radio("Data source", ["Synthetic data", "Upload CSV"])
if source == "Synthetic data":
    n_samples = st.sidebar.slider("Number of samples", 100, 2000, 400, 50)
    seed = st.sidebar.number_input("Random seed", value=42, step=1)
    df_raw = generate_data(n_samples, int(seed))
else:
    up = st.sidebar.file_uploader("Upload facility_reservations.csv", type="csv")
    seed = st.sidebar.number_input("Random seed", value=42, step=1)
    if up is None:
        st.info("Upload a CSV with the columns: Reservation_ID, User_Category, Facility_Type, "
                "Lead_Time_Days, Participants, Actual_Attendance, Usage_Status, Cancellation_Days_Before.")
        st.stop()
    df_raw = pd.read_csv(up)

st.sidebar.subheader("Model")
max_depth = st.sidebar.slider("Decision tree max depth", 1, 10, 4)
test_size = st.sidebar.slider("Test set size", 0.1, 0.5, 0.20, 0.05)

df, log = clean_data(df_raw)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
st.title("🏛️ Facility Reservation Analysis")
st.caption("Data collection → cleaning → EDA → decision-tree prediction of reservation outcome")

tab1, tab2, tab3, tab4 = st.tabs([
    "Part I: Data Overview", "Part II: Cleaning & Features",
    "Part III: EDA", "Part IV: Predictive Modeling"])

# ---- Part I -----------------------------------------------------------------
with tab1:
    st.subheader("Raw dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df_raw.shape[0])
    c2.metric("Columns", df_raw.shape[1])
    valid = (df_raw['Lead_Time_Days'] >= 0) & (df_raw['Participants'] > 0)
    c3.metric("Invalid records", int((~valid).sum()), help="Lead_Time_Days < 0 or Participants <= 0")

    st.markdown("**First 10 records**")
    st.dataframe(df_raw.head(10), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Variable types**")
        st.dataframe(df_raw.dtypes.astype(str).rename("dtype"), use_container_width=True)
    with col_b:
        st.markdown("**Summary statistics**")
        st.dataframe(df_raw.describe(include='all').T, use_container_width=True)

    with st.expander("Task 3: Data collection (SQL query & validation)"):
        st.code("""SELECT Reservation_ID, User_Category, Facility_Type,
       Lead_Time_Days, Participants, Actual_Attendance,
       Usage_Status, Cancellation_Days_Before
FROM Facility_Reservations
WHERE Reservation_Date >= '2025-01-01';""", language="sql")
        st.write("**Validation at point of entry:** `Lead_Time_Days >= 0` and `Participants > 0`.")

    with st.expander("Data dictionary"):
        st.table(pd.DataFrame({
            "Variable": ["Reservation_ID", "User_Category", "Facility_Type", "Lead_Time_Days",
                         "Participants", "Actual_Attendance", "Usage_Status", "Cancellation_Days_Before"],
            "Description": ["Unique reservation identifier", "Category of the user",
                            "Type of facility reserved", "Days between booking and reservation",
                            "Number of participants", "Number of actual attendees",
                            "Used, Cancelled, or No-Show", "Days before reservation when cancelled"],
        }))

    with st.expander("Variable roles (Task 2)"):
        st.markdown("""
- **Predictor candidates:** User_Category, Facility_Type, Lead_Time_Days, Participants, Lead_Time_Category
- **Target variable:** Reservation_Outcome (derived: Completed vs Not Completed)
- **Categorical:** User_Category, Facility_Type
- **Numerical:** Lead_Time_Days, Participants
""")

# ---- Part II ----------------------------------------------------------------
with tab2:
    st.subheader("Data cleaning")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original rows", len(df_raw))
    c2.metric("Duplicates removed", log['dup_count'])
    c3.metric("Invalid lead times", log['invalid_lead'])
    c4.metric("Invalid participants", log['invalid_part'])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Missing values – before**")
        st.dataframe(log['missing_before'].rename("missing"), use_container_width=True)
        st.markdown("**User categories – before standardization**")
        st.dataframe(log['cat_before'], use_container_width=True)
    with col_b:
        st.markdown("**Missing values – after**")
        st.dataframe(log['missing_after'].rename("missing"), use_container_width=True)
        st.markdown("**User categories – after standardization**")
        st.dataframe(log['cat_after'], use_container_width=True)

    st.info("Invalid Lead_Time_Days (<0 or >180) and Participants (≤0) were set to NaN and imputed "
            "with the median. Actual_Attendance and Cancellation_Days_Before NaNs are left as-is "
            "(they are structurally missing).")

    st.subheader("Feature engineering")
    st.markdown("""
- **Lead_Time_Category:** ≤2 days = *Last Minute*, ≤7 = *Short Lead*, ≤14 = *Medium Lead*, else *Early Booking*
- **Reservation_Outcome:** *Completed* if Usage_Status = Used, otherwise *Not Completed*
- **Attendance_Ratio:** Actual_Attendance / Participants (0 if either is missing or zero)
""")
    st.write(f"Cleaned dataset shape: **{df.shape[0]} × {df.shape[1]}**")
    st.dataframe(df.head(10), use_container_width=True)
    st.download_button("⬇️ Download cleaned CSV", df.to_csv(index=False).encode(),
                       "cleaned_facility_reservations.csv", "text/csv")

# ---- Part III ---------------------------------------------------------------
with tab3:
    st.subheader("Exploratory data analysis")
    fac_counts = df['Facility_Type'].value_counts()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**1. Reservations by facility type**")
        st.dataframe(fac_counts.rename("count"), use_container_width=True)
    with col_b:
        st.markdown("**2. Outcome % by user category**")
        user_outcome = pd.crosstab(df['User_Category'], df['Reservation_Outcome'], normalize='index') * 100
        st.dataframe(user_outcome.round(2), use_container_width=True)

    st.markdown("**3. Outcome % by lead-time category**")
    lead_outcome = pd.crosstab(df['Lead_Time_Category'], df['Reservation_Outcome'], normalize='index') * 100
    st.dataframe(lead_outcome.round(2), use_container_width=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(data=df, x='Facility_Type', hue='Facility_Type', palette='viridis',
                  ax=axes[0], order=fac_counts.index, legend=False)
    axes[0].set_title('Figure 1: Total Reservation Count by Facility Type')
    axes[0].set_xlabel('Facility Type')
    axes[0].set_ylabel('Number of Reservations')

    lp = pd.crosstab(df['Lead_Time_Category'], df['Reservation_Outcome'], normalize='index').reset_index()
    lp_m = pd.melt(lp, id_vars='Lead_Time_Category', value_vars=['Completed', 'Not Completed'])
    sns.barplot(data=lp_m, x='Lead_Time_Category', y='value', hue='Reservation_Outcome',
                palette='Set2', ax=axes[1])
    axes[1].set_title('Figure 2: Reservation Outcome Proportions by Lead Time')
    axes[1].set_xlabel('Lead Time Category')
    axes[1].set_ylabel('Proportion')
    plt.tight_layout()
    st.pyplot(fig)

# ---- Part IV ----------------------------------------------------------------
with tab4:
    st.subheader("Decision tree classifier")
    clf, feat_names, metrics, cm_df, report, importances = train_model(
        df, max_depth, test_size, int(seed))

    cols = st.columns(4)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.4f}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Confusion matrix**")
        st.dataframe(cm_df, use_container_width=True)
        st.markdown("**Classification report**")
        st.dataframe(report.round(3), use_container_width=True)
    with col_b:
        st.markdown("**Top feature importances**")
        top = importances.head(8)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=top.values, y=top.index, hue=top.index, palette='mako', legend=False, ax=ax2)
        ax2.set_xlabel("Importance")
        plt.tight_layout()
        st.pyplot(fig2)

    with st.expander("Visualize the decision tree"):
        fig3, ax3 = plt.subplots(figsize=(20, 9))
        plot_tree(clf, feature_names=feat_names, class_names=list(clf.classes_),
                  filled=True, rounded=True, fontsize=8, ax=ax3)
        st.pyplot(fig3)
