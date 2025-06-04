import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Bank Customer Churn Prediction.csv")
    return df

# Preprocessing
def preprocess_data(data):
    data = data.copy()
    
    # Before preprocessing info
    before_info = {
        'missing_values': data.isnull().sum().sum(),
        'duplicates': data.duplicated().sum(),
        'shape': data.shape
    }
    
    # Handle missing values (if any)
    data = data.fillna(data.mean(numeric_only=True))
    
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Handle outliers using IQR method for numerical columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if col not in ['customer_id', 'churn']:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
            data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
    
    # Label encoding for categorical variables
    label_encoder = LabelEncoder()
    if 'gender' in data.columns:
        data['gender'] = label_encoder.fit_transform(data['gender'])
    if 'country' in data.columns:
        data['country'] = label_encoder.fit_transform(data['country'])
    
    # Remove customer_id
    if 'customer_id' in data.columns:
        data = data.drop('customer_id', axis=1)
    
    # After preprocessing info
    after_info = {
        'missing_values': data.isnull().sum().sum(),
        'duplicates': data.duplicated().sum(),
        'shape': data.shape
    }
    
    return data, before_info, after_info

# Model Training Function
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_train, y_train)
    return model

# KMeans Clustering Function
def perform_clustering(data, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return kmeans, clusters

# Elbow Method for optimal clusters
def plot_elbow_method(data, max_k=10):
    inertias = []
    K_range = range(1, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(K_range, inertias, 'bo-')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method for Optimal Number of Clusters')
    ax.grid(True)
    return fig

# Silhouette Score Analysis
def plot_silhouette_scores(data, max_k=10):
    silhouette_scores = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(K_range, silhouette_scores, 'ro-')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Average Silhouette Score')
    ax.set_title('Silhouette Analysis for Optimal Number of Clusters')
    ax.grid(True)
    return fig
def boxplot_outlier_analysis(data):
    plt.figure(figsize=(30, 10))  # Set plot size
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    
    # Plot boxplot for all numerical columns
    data.select_dtypes(include=[np.number]).boxplot()
    plt.title("Boxplot for Outlier Analysis")  # Title for the plot
    plt.ylabel('Value')  # Label for y-axis
    plt.xlabel('Columns')  # Label for x-axis
    plt.show()
# Main Dashboard App
def main():
    st.title("Bank Customer Churn Dashboard")
    
    # Load data
    df = load_data()
    
    # Tabs
    tabs = st.tabs(["Business Understanding", 
                    "Data Understanding", 
                    "Exploratory Data Analysis",
                    "Data Preprocessing", 
                    "Modelling & Evaluation", 
                    "Customer Churn Prediction"])
    
    # Tab 1: Business Understanding
    with tabs[0]:
        st.header("Business Understanding")
        st.write("""
        ### Latar Belakang
        Di tengah persaingan yang semakin ketat di industri perbankan, kemampuan untuk mempertahankan loyalitas nasabah menjadi salah satu indikator utama kesuksesan suatu institusi keuangan. Customer churn, atau kehilangan nasabah, adalah masalah yang umum terjadi jika bank tidak dapat memberikan layanan yang sesuai dengan harapan pelanggan.
        
        ### Dampak Customer Churn
        - **Penurunan Pendapatan**: Kehilangan nasabah berarti kehilangan sumber pendapatan
        - **Reputasi Perusahaan**: Dapat memengaruhi citra dan kepercayaan publik
        - **Biaya Akuisisi**: Biaya mendapatkan nasabah baru lebih tinggi daripada mempertahankan yang ada
        
        ### Tujuan Analisis
        Melalui analisis dataset Bank Customer Churn, penelitian ini berupaya untuk:
        - **Mengidentifikasi pola dan karakteristik** nasabah yang cenderung melakukan churn
        - **Menggunakan teknik analisis data**, seperti clustering dan regresi logistik, untuk memprediksi nasabah yang berisiko tinggi churn
        - **Memberikan rekomendasi strategis** untuk bank dalam menangani churn dan meningkatkan loyalitas nasabah
        
        ### Manfaat
        Dengan memanfaatkan wawasan yang diperoleh melalui eksplorasi dan analisis data, bank dapat:
        - Mengoptimalkan alokasi sumber daya
        - Meningkatkan pengalaman pelanggan
        - Mengurangi tingkat churn secara signifikan
        """)

    # Tab 2: Data Understanding
    with tabs[1]:
        st.header("Data Understanding")
        st.write("### Preview Dataset:")
        st.write(df.head())
        
        st.write("### Statistik Deskriptif:")
        st.write(df.describe())
        
        st.write("### Cek Missing Values:")
        st.write(df.isnull().sum())
        
        st.subheader("### Cek Duplikasi Data")
        st.write(f"Jumlah data duplikat: {df.duplicated().sum()}")

    # Tab 3: Exploratory Data Analysis
    with tabs[2]:
        st.header("Exploratory Data Analysis")
        
        # Churn Distribution by various factors
        st.subheader("Churn Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn by Gender
            if 'gender' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                churn_gender = pd.crosstab(df['gender'], df['churn'], normalize='index') * 100
                churn_gender.plot(kind='bar', ax=ax)
                ax.set_title('Churn Rate by Gender')
                ax.set_ylabel('Percentage')
                ax.legend(['No Churn', 'Churn'])
                plt.xticks(rotation=45)
                st.pyplot(fig)
                st.write("*Analisis tingkat churn berdasarkan jenis kelamin nasabah*")
        
        with col2:
            # Churn by Country
            if 'country' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                churn_country = pd.crosstab(df['country'], df['churn'], normalize='index') * 100
                churn_country.plot(kind='bar', ax=ax)
                ax.set_title('Churn Rate by Country')
                ax.set_ylabel('Percentage')
                ax.legend(['No Churn', 'Churn'])
                plt.xticks(rotation=45)
                st.pyplot(fig)
                st.write("*Perbandingan tingkat churn antar negara*")
        
        # Age distribution analysis
        st.subheader("Age Distribution Analysis")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Age distribution by churn
        df[df['churn'] == 0]['age'].hist(ax=ax1, alpha=0.7, label='No Churn', bins=30)
        df[df['churn'] == 1]['age'].hist(ax=ax1, alpha=0.7, label='Churn', bins=30)
        ax1.set_title('Age Distribution by Churn Status')
        ax1.set_xlabel('Age')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # QQ Plot for Age
        stats.probplot(df['age'], dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot for Age Distribution')
        st.pyplot(fig)
        st.write("*Distribusi usia menunjukkan pola churn berdasarkan kelompok usia tertentu*")
        
        
        # Correlation Matrix
        st.subheader("Feature Correlation Analysis")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)
        st.write("*Matriks korelasi menunjukkan hubungan antar variabel numerik*")

    # Tab 4: Data Preprocessing
    with tabs[3]:
        st.header("Data Preprocessing")
    
        st.write("### Langkah-langkah Preprocessing:")
        st.write("""
        1. **Penanganan Missing Values**: Imputation menggunakan mean untuk variabel numerik.
        2. **Deteksi dan Treatment Outliers**: Menggunakan metode IQR (Interquartile Range). 
        3. **Eliminasi Data Duplikat**: Menghapus baris yang duplikat.
        """)
        # Perform preprocessing
        preprocessed_data, _, _ = preprocess_data(df)
        fig, ax = plt.subplots(figsize=(15, 8))
        preprocessed_data.select_dtypes(include=[np.number]).boxplot(ax=ax, rot=90)
        ax.set_title("Boxplot of Numerical Features After Preprocessing")
        ax.set_ylabel("Values")
        st.pyplot(fig)
        
        

    # Tab 5: Modelling & Evaluation
    with tabs[4]:
        st.header("Modelling & Evaluation")
        
        # Splitting dataset into train and test sets
        preprocessed_data, _, _ = preprocess_data(df)
        X = preprocessed_data.drop('churn', axis=1)
        y = preprocessed_data['churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scaling features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Logistic Regression Model
        st.subheader("Logistic Regression Model for Churn Prediction")
        log_reg_model = train_logistic_regression(X_train_scaled, y_train)
        y_pred = log_reg_model.predict(X_test_scaled)
        y_pred_proba = log_reg_model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy of Logistic Regression Model:** {accuracy*100:.2f}%")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        # ROC Curve Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC Curve
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax1.legend(loc="lower right")
        ax1.grid(True)
        
        # Confusion Matrix Heatmap
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title('Confusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        st.pyplot(fig)
        st.write("*ROC Curve menunjukkan performa model dalam membedakan kelas churn dan non-churn*")
        
        st.write("### Classification Report:")
        st.text(classification_report(y_test, y_pred))
        
        # Feature Importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': abs(log_reg_model.coef_[0])
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature', ax=ax)
        ax.set_title('Top 10 Most Important Features')
        ax.set_xlabel('Importance (Absolute Coefficient Value)')
        st.pyplot(fig)
        st.write("*Fitur dengan koefisien tertinggi memiliki pengaruh paling besar terhadap prediksi churn*")
        
        # KMeans Clustering Analysis
        st.subheader("KMeans Clustering for Customer Segmentation")
        
        # Elbow Method
        st.write("### Elbow Method for Optimal Clusters")
        elbow_fig = plot_elbow_method(X_train_scaled, max_k=10)
        st.pyplot(elbow_fig)
        st.write("*Metode Elbow membantu menentukan jumlah cluster optimal dengan melihat titik 'siku' pada grafik*")
        
        # Silhouette Analysis
        st.write("### Silhouette Score Analysis")
        silhouette_fig = plot_silhouette_scores(X_train_scaled, max_k=10)
        st.pyplot(silhouette_fig)
        st.write("*Silhouette Score mengukur seberapa baik setiap titik data cocok dengan cluster-nya*")
        
        # Perform clustering with user input
        n_clusters = st.slider("Select number of clusters:", 2, 10, 5)
        kmeans, clusters = perform_clustering(X_train_scaled, n_clusters=n_clusters)
        
        # Silhouette score for selected clusters
        silhouette_avg = silhouette_score(X_train_scaled, clusters)
        st.write(f"**Average Silhouette Score for {n_clusters} clusters:** {silhouette_avg:.3f}")
        
        st.write("### Cluster Centers:")
        cluster_centers_df = pd.DataFrame(
            scaler.inverse_transform(kmeans.cluster_centers_), 
            columns=X.columns
        )
        st.write(cluster_centers_df)
        
        # Add cluster labels to original data
        clustered_df = X.copy()
        clustered_df['Cluster'] = kmeans.predict(scaler.transform(X))
        clustered_df['Churn'] = y
        
        st.write("### Customer Segmentation Results:")
        cluster_summary = clustered_df.groupby('Cluster').agg({
            'age': 'mean',
            'balance': 'mean',
            'credit_score': 'mean',
            'Churn': ['count', 'mean']
        }).round(2)
        st.write(cluster_summary)
        
        # Visualizing Clusters
        st.write("### Cluster Visualization:")
        
        # Create 2D visualization using first two principal components or selected features
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Age vs Balance
        scatter = ax1.scatter(clustered_df['age'], clustered_df['balance'], 
                             c=clustered_df['Cluster'], cmap='viridis', alpha=0.6)
        ax1.set_xlabel("Age")
        ax1.set_ylabel("Balance")
        ax1.set_title("Customer Clusters: Age vs Balance")
        plt.colorbar(scatter, ax=ax1)
        
        # Credit Score vs Balance
        scatter2 = ax2.scatter(clustered_df['credit_score'], clustered_df['balance'], 
                              c=clustered_df['Cluster'], cmap='viridis', alpha=0.6)
        ax2.set_xlabel("Credit Score")
        ax2.set_ylabel("Balance")
        ax2.set_title("Customer Clusters: Credit Score vs Balance")
        plt.colorbar(scatter2, ax=ax2)
        
        st.pyplot(fig)
        st.write("*Visualisasi cluster membantu memahami karakteristik setiap segmen pelanggan*")
        
        # Churn rate by cluster
        st.write("### Churn Rate by Cluster:")
        churn_by_cluster = clustered_df.groupby('Cluster')['Churn'].agg(['count', 'sum', 'mean']).round(3)
        churn_by_cluster.columns = ['Total_Customers', 'Churned_Customers', 'Churn_Rate']
        churn_by_cluster['Churn_Percentage'] = (churn_by_cluster['Churn_Rate'] * 100).round(2)
        st.write(churn_by_cluster)
        
        # Churn rate visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=churn_by_cluster.index, y=churn_by_cluster['Churn_Percentage'], ax=ax)
        ax.set_title('Churn Rate by Customer Cluster')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Churn Rate (%)')
        for i, v in enumerate(churn_by_cluster['Churn_Percentage']):
            ax.text(i, v + 0.5, f'{v}%', ha='center', va='bottom')
        st.pyplot(fig)
        st.write("*Setiap cluster memiliki tingkat churn yang berbeda, membantu identifikasi segmen berisiko tinggi*")


    # Tab 5: Customer Churn Prediction
    with tabs[5]:
        st.header("Customer Churn Prediction")
        st.write("Use the form below to predict the churn for a specific customer.")
        
        credit_score = st.number_input("Credit Score", step=1, min_value=0)
        country = st.selectbox("Country", options=["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", options=["Male", "Female"])
        age = st.number_input("Age", step=1, min_value=0)
        tenure = st.number_input("Tenure", step=1, min_value=0)
        balance = st.number_input("Balance", step=1, min_value=0)
        products_number = st.number_input("Number of Products", step=1, min_value=0)
        credit_card = st.selectbox("Has Credit Card?", options=["Yes", "No"])
        active_member = st.selectbox("Active Member?", options=["Yes", "No"])
        estimated_salary = st.number_input("Estimated Salary", step=1, min_value=0)
        
        if st.button("Predict Churn"):
            # Process input data
            input_data = {
                "credit_score": credit_score,
                "country": 0 if country == "France" else 1 if country == "Germany" else 2,
                "gender": 0 if gender == "Male" else 1,
                "age": age,
                "tenure": tenure,
                "balance": balance,
                "products_number": products_number,
                "credit_card": 1 if credit_card == "Yes" else 0,
                "active_member": 1 if active_member == "Yes" else 0,
                "estimated_salary": estimated_salary
            }
            input_df = pd.DataFrame([input_data])
            input_df = scaler.transform(input_df)
            prediction = log_reg_model.predict(input_df)
            churn_result = "Yes" if prediction[0] == 1 else "No"
            st.write(f"**Prediction: Will the customer churn?** {churn_result}")

if __name__ == "__main__":
    main()