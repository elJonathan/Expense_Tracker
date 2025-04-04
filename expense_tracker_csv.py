import streamlit as st
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from dotenv import load_dotenv
import os
import tempfile
from fpdf import FPDF
from datetime import datetime

#This is Just a comment

# Load environment variables from .env file
load_dotenv()

# ---------- Stream Settings ----------
st.set_page_config(page_title="Enhanced Expense Tracker", layout="wide")
st.markdown("""
    <style>
    body { font-family: 'Times New Roman', serif; }
    </style>
""", unsafe_allow_html=True)

# ---------- DB Connection ----------
def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASS", "root"),
        database=os.getenv("DB_NAME", "expense_tracker"),
        connection_timeout=5
    )

def get_connection_with_retry(retries=3, delay=2):
    import time
    for _ in range(retries):
        try:
            conn = get_connection()
            if conn.is_connected():
                return conn
        except mysql.connector.Error:
            time.sleep(delay)
    st.error("Could not connect to the database.")
    st.stop()

# ---------- DB Operations ----------
def add_expense(date, category, amount, description, payment_mode):
    with get_connection_with_retry() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO expenses (date, category, amount, description, payment_mode)
            VALUES (%s, %s, %s, %s, %s)
        """, (date, category, amount, description, payment_mode))
        conn.commit()

def get_expenses():
    with get_connection_with_retry() as conn:
        df = pd.read_sql("SELECT id, date, category, amount, description, payment_mode FROM expenses ORDER BY date DESC", conn)
        df['date'] = pd.to_datetime(df['date']).dt.date  # Remove time portion from date column
        return df

def delete_expense(expense_id):
    with get_connection_with_retry() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM expenses WHERE id = %s", (expense_id,))
        conn.commit()

def update_expense(expense_id, date, category, amount, description, payment_mode):
    with get_connection_with_retry() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE expenses
            SET date = %s, category = %s, amount = %s, description = %s, payment_mode = %s
            WHERE id = %s
        """, (date, category, amount, description, payment_mode, expense_id))
        conn.commit()

# ---------- PDF Report ----------
def generate_pdf_report(df, pie_fig=None, hist_fig=None, heatmap_fig=None):
    df = df.drop(columns=["id"])  # Remove ID column for PDF export
    class PDF(FPDF):
        def footer(self):
            self.set_y(-15)
            self.set_font("DejaVu", "", 8)
            self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    pdf = PDF()
    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("DejaVu", "", 24)
    pdf.cell(0, 20, "Expense Report Summary", ln=True, align="C")
    pdf.set_font("DejaVu", "", 14)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.cell(0, 10, "Prepared by: Jonathan Jackson", ln=True, align="C")
    pdf.cell(0, 10, "Prepared using Enhanced Expense Tracker", ln=True, align="C")
    # Add logo to bottom center of first page
    logo_path = os.path.join(os.path.dirname(__file__), "IMG-20240519- 001WA0016.jpg")
    if os.path.exists(logo_path):
        pdf.image(logo_path, x=pdf.w / 2 - 20, y=pdf.h - 40, w=40)

    pdf.add_page()
    pdf.set_font("DejaVu", "", 16)
    pdf.cell(0, 10, "Summary", ln=True)
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 10, f"Total Expenses: ₹{df['amount'].sum():,.2f}", ln=True)
    pdf.cell(0, 10, f"Total Records: {len(df)}", ln=True)

    chart_tempfiles = []

    def insert_chart(fig, label):
        if fig:
            pdf.add_page()
            tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig.savefig(tmpfile.name, format="png", bbox_inches="tight")
            tmpfile.close()
            chart_tempfiles.append(tmpfile.name)
            pdf.set_font("DejaVu", "", 14)
            pdf.cell(0, 10, label, ln=True)
            pdf.image(tmpfile.name, x=(pdf.w - 180) / 2, w=180)

    chart_titles = [
        "Monthly Spending Trend",
        "Payment Mode Breakdown",
        "Top Spending Categories",
        "Amount Distribution",
        "Expenses by Category",
        "Category Spending Over Time"
    ]

    if isinstance(pie_fig, list):
        for fig, title in zip(pie_fig, chart_titles):
            insert_chart(fig, title)

    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    pdf.output(pdf_path)

    for tmp_path in chart_tempfiles:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return pdf_path

# ---------- UI ----------
top_row = st.columns([6, 1])
with top_row[0]:
    st.title("💸 Enhanced Expense Tracker")
with top_row[1]:
    currency_symbol = st.selectbox("💱", ["₹", "$", "€"], index=0, label_visibility="collapsed")
    

# Utility to format currency
format_currency = lambda amount: f"{currency_symbol}{amount:,.2f}"
tab1, tab2 = st.tabs(["➕ Add Expense", "📊 View Reports"])

with tab1:
    st.subheader("Add a New Expense")
    with st.form("expense_form"):
        col1, col2 = st.columns(2)
        with col1:
            expense_date = st.date_input("Date", value=date.today())
            category = st.selectbox("Category", ["Food", "Transport", "Bills", "Shopping", "Entertainment", "Other"])
            payment_mode = st.selectbox("Payment Mode", ["Cash", "Credit Card", "Debit Card", "UPI", "Other"])
        with col2:
            amount = st.number_input("Amount", min_value=0.0, format="%.2f")
            description = st.text_input("Description")

        if st.form_submit_button("Add Expense"):
            add_expense(expense_date, category, amount, description, payment_mode)
            st.success("✅ Expense added successfully!")

    st.markdown("---")
    st.subheader("📥 Upload CSV File")
    st.markdown("⚠️ Uploaded data will remain in the database **until you delete it manually**.")
    delete_uploaded = st.button("🗑️ Delete All Uploaded Data")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader_main")
    if delete_uploaded:
        with get_connection_with_retry() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM expenses")
            conn.commit()
        st.success("🧹 All uploaded expenses deleted from database!")

        
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            required_columns = {"date", "category", "amount", "description", "payment_mode"}
            if required_columns.issubset(uploaded_df.columns):
                uploaded_df["date"] = pd.to_datetime(uploaded_df["date"]).dt.date
                existing = get_expenses()
                for _, row in uploaded_df.iterrows():
                    if not ((existing["date"] == row["date"]) &
                            (existing["category"] == row["category"]) &
                            (existing["amount"] == row["amount"]) &
                            (existing["description"] == row["description"]) &
                            (existing["payment_mode"] == row["payment_mode"])).any():
                        add_expense(row["date"], row["category"], row["amount"], row["description"], row["payment_mode"])
                st.success("✅ CSV data uploaded successfully!")
            else:
                st.error("CSV must contain: date, category, amount, description, payment_mode")
        except Exception as e:
            st.error(f"Error processing file: {e}")
            add_expense(expense_date, category, amount, description, payment_mode)
            st.success("✅ Expense added successfully!")

with tab2:
    df_all = get_expenses()
    st.metric(label="💰 Total Expenses", value=format_currency(df_all['amount'].sum()))
    if df_all.empty:
        st.warning("No expenses to display.")
        st.stop()

    with st.expander("🔍 Global Filters"):
        categories = df_all["category"].unique()
        selected = st.multiselect("Category", categories, default=list(categories))
        min_date = df_all['date'].min()
        max_date = df_all['date'].max()
        date_range = st.date_input("Date Range", [min_date, max_date])
        keyword = st.text_input("Search Description Keyword")
        clear = st.button("🧹 Clear All Filters")

        if clear:
            selected = list(categories)
            date_range = [min_date, max_date]
            keyword = ""

        filtered_df = df_all[df_all["category"].isin(selected)]
        if date_range and len(date_range) == 2 and date_range != [min_date, max_date]:
            filtered_df = filtered_df[(filtered_df["date"] >= date_range[0]) & (filtered_df["date"] <= date_range[1])]
        if keyword:
            filtered_df = filtered_df[filtered_df["description"].str.contains(keyword, case=False)]

    subtab1, subtab2 = st.tabs(["📜 Expense History", "📈 Analysis & Visualizations"])

    with subtab1:
        st.subheader("📜 Expense History")
        edited_df = st.data_editor(
        filtered_df.drop(columns=["id"]),
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "amount": st.column_config.NumberColumn("Amount ($)", format="%.2f"),
                "date": st.column_config.DateColumn("Date")
            },
            key="editable_table"
        )

        if st.button("💾 Save Changes"):
            if not edited_df.empty:
                for i, row in edited_df.iterrows():
                    original_id = int(filtered_df.iloc[i]["id"])
                    update_expense(
                original_id,
                row["date"],
                row["category"],
                float(row["amount"]),  # Ensure float for MySQL compatibility
                row["description"],
                row["payment_mode"]
                )
            st.success("✅ All changes saved to the database!")

    with subtab2:
        st.subheader("📈 Analysis & Visualizations")
        sns.set(style="darkgrid")  # Changed to dark theme
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.style.use('dark_background')
        plt.rcParams.update({
          'axes.facecolor': 'none',  # Transparent axes
          'figure.facecolor': 'none',  # Transparent figure
          'axes.grid': False,  # Remove gridlines
          'axes.edgecolor': 'navy',
          'axes.labelcolor': 'lightblue',
          'xtick.color': 'lightblue',
          'ytick.color': 'lightblue',
          'text.color': 'lightblue',
          'axes.titlecolor': 'lightblue'
        })
          
        # Use dark background for all plots
        filtered_df["date"] = pd.to_datetime(filtered_df["date"])
        filtered_df["weekday"] = filtered_df["date"].dt.day_name()
        filtered_df["month"] = filtered_df["date"].dt.to_period("M")
        fig_size = (5, 3)

        fig1, ax1 = plt.subplots(figsize=fig_size, facecolor='none')
        ax1.spines[:].set_color('navy')
        ax1.tick_params(colors='white')
        ax1.yaxis.label.set_color('white')
        ax1.xaxis.label.set_color('white')
        ax1.title.set_color('white')

        if (filtered_df['date'].max() - filtered_df['date'].min()).days <= 31:
            daily_data = filtered_df.groupby('date')["amount"].sum()
            daily_labels = daily_data.index.strftime('%d %b')
            step = max(1, len(daily_labels) // 10)
            ax1.set_xticks(range(0, len(daily_labels), step))
            ax1.set_xticklabels(daily_labels[::step], rotation=45, ha='right')
            ax1.plot(daily_labels, daily_data.values, marker='o', color='red')
        else:
            monthly_data = filtered_df.groupby("month")["amount"].sum()
            monthly_data.index = monthly_data.index.to_timestamp()
            month_labels = monthly_data.index.strftime('%b')
            ax1.plot(month_labels, monthly_data.values, marker='o', color='red')

        import matplotlib.ticker as ticker
        temp_df = filtered_df.copy()
        temp_df['date'] = pd.to_datetime(temp_df['date'])
        temp_df.set_index('date', inplace=True)
        max_y = temp_df['amount'].resample('M').sum().max()
        ax1.set_ylim(0, max_y + 50000)
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(100000))
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / 100000:.1f}'))
        ax1.set_ylabel("(in 100K)")
        ax1.set_title("Monthly Spending Trend")
        ax1.set_facecolor('none')
        st.pyplot(fig1, use_container_width=True, transparent=True)

        if not filtered_df.empty and filtered_df["payment_mode"].nunique() > 0:
            fig2, ax2 = plt.subplots(figsize=fig_size, facecolor='none')
            ax2.set_facecolor('none')
            ax2.set_title("Payment Mode Breakdown", color='lightblue')
            filtered_df.groupby("payment_mode")["amount"].sum().plot(kind="pie", autopct="%1.1f%%", ax=ax2, textprops={'color':'black'}, wedgeprops={'edgecolor':'navy'}, colors=sns.color_palette('dark').as_hex())
            for text in ax2.texts:
                text.set_color('lightblue')
            ax2.set_title("Payment Mode Breakdown")
            ax2.set_ylabel("")
            st.pyplot(fig2, use_container_width=True, transparent=True)
        else:
            st.info("No data available to display Payment Mode Breakdown.")  

        fig3, ax3 = plt.subplots(figsize=fig_size, facecolor='none')
        ax3.spines[:].set_color('navy')
        ax3.tick_params(colors='lightblue')
        ax3.yaxis.label.set_color('lightblue')
        ax3.xaxis.label.set_color('lightblue')
        ax3.title.set_color('lightblue')
        filtered_df.groupby("category")["amount"].sum().sort_values().plot(kind="barh", ax=ax3, color=sns.color_palette('dark').as_hex(), edgecolor='navy')
        ax3.set_title("Top Spending Categories")
        ax3.set_facecolor('none')  # Remove background grid
        ax3.grid(False)
        st.pyplot(fig3, use_container_width=True, transparent=True)

        fig4, ax4 = plt.subplots(figsize=fig_size, facecolor='none')
        ax4.spines[:].set_color('navy')
        ax4.tick_params(colors='lightblue')
        ax4.yaxis.label.set_color('lightblue')
        ax4.xaxis.label.set_color('lightblue')
        ax4.title.set_color('lightblue')
        filtered_df["amount"].plot(kind="hist", bins=20, ax=ax4, edgecolor='navy', color=sns.color_palette('dark')[1])
        ax4.set_title("Amount Distribution")
        ax4.set_facecolor('none')  # Remove background grid
        ax4.grid(False)
        ax4.set_xlabel("Amount")
        st.pyplot(fig4, use_container_width=True, transparent=True)

        if not filtered_df.empty and filtered_df["category"].nunique() > 0:
            fig6, ax6 = plt.subplots(figsize=fig_size, facecolor='none')
            ax6.set_facecolor('none')
            ax6.spines[:].set_color('navy')
            ax6.tick_params(colors='lightblue', labelrotation=0)
            ax6.set_xticklabels(ax6.get_xticklabels(), rotation=0, ha='center')
            ax6.yaxis.label.set_color('lightblue')
            ax6.xaxis.label.set_color('lightblue')
            ax6.title.set_color('lightblue')
            monthly_cat_data = filtered_df.groupby(["month", "category"])["amount"].sum().unstack().fillna(0)
            monthly_cat_data = monthly_cat_data.sort_index()
            monthly_cat_data.index = monthly_cat_data.index.to_timestamp()
            monthly_cat_data.index.name = None
            monthly_cat_data.index = monthly_cat_data.index.strftime('%b')
            monthly_cat_data.plot(kind="bar", stacked=True, ax=ax6, edgecolor='navy', color=sns.color_palette('dark').as_hex(), width=0.8)
            ax6.set_title("Category Spending Over Time")
            ax6.set_facecolor('none')
            ax6.grid(False)
            ax6.set_xlabel("")
            ax6.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
            import matplotlib.ticker as ticker
            ax6.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/100000:.1f}'))
            ax6.set_ylabel("(in 100K)")
            st.pyplot(fig6, use_container_width=True, transparent=True)
        else:
            st.info("No data available to display Category Spending Over Time.")
        # Export Excel
        if not filtered_df.empty:
            excel_filename = "filtered_expenses.xlsx"
            filtered_df_copy = filtered_df.drop(columns=["id"]).copy()
            filtered_df_copy['date'] = pd.to_datetime(filtered_df_copy['date']).dt.strftime('%Y-%m-%d')
            filtered_df_copy.to_excel(excel_filename, index=False)
            with open(excel_filename, "rb") as file:
                            st.download_button("📤 Download Filtered Data as Excel", file, excel_filename)

        # Export PDF
        if not filtered_df.empty:
            pdf_path = generate_pdf_report(filtered_df, pie_fig=[fig1, fig2, fig3, fig4, fig6])
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download PDF Report", data=f, file_name="expense_summary_report.pdf", mime="application/pdf")
