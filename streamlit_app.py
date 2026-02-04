"""
Real Estate Investment Advisor - Streamlit Application
Interactive web app for property investment predictions and analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .good-investment {
        color: #2ca02c;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .bad-investment {
        color: #d62728;
        font-weight: bold;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load processed data"""
    try:
        df = pd.read_csv("data/processed/housing_data_processed.csv")
        return df
    except:
        st.error("Error loading data. Please run data preprocessing first.")
        return None

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🏠 Real Estate Investment Advisor</div>', unsafe_allow_html=True)
    st.markdown("### Predict Property Profitability & Future Value")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🔮 Predict", "📊 Analytics Dashboard", "📈 Market Insights", "📉 EDA Results", "ℹ️ About"])
    
    if page == "🔮 Predict":
        prediction_page(df)
    elif page == "📊 Analytics Dashboard":
        analytics_page(df)
    elif page == "📈 Market Insights":
        market_insights_page(df)
    elif page == "📉 EDA Results":
        eda_results_page(df)
    else:
        about_page()

def prediction_page(df):
    """Property prediction page"""
    st.markdown('<div class="sub-header">Property Investment Prediction</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Property Details")
        
        # Get unique values for dropdowns
        states = sorted(df['State'].unique())
        property_types = sorted(df['Property_Type'].unique())
        furnished_status = sorted(df['Furnished_Status'].unique())
        facing_options = sorted(df['Facing'].unique())
        
        # Input form
        state = st.selectbox("State", states)
        cities = sorted(df[df['State'] == state]['City'].unique())
        city = st.selectbox("City", cities)
        
        property_type = st.selectbox("Property Type", property_types)
        bhk = st.slider("BHK", 1, 5, 3)
        size_sqft = st.number_input("Size (Sq Ft)", min_value=500, max_value=5000, value=1500, step=100)
        price_lakhs = st.number_input("Current Price (Lakhs)", min_value=10.0, max_value=500.0, value=100.0, step=5.0)
        
        furnished = st.selectbox("Furnished Status", furnished_status)
        facing = st.selectbox("Facing Direction", facing_options)
        
        age = st.slider("Age of Property (Years)", 0, 30, 5)
        floor_no = st.slider("Floor Number", 0, 20, 5)
        total_floors = st.slider("Total Floors", 1, 25, 10)
        
        nearby_schools = st.slider("Nearby Schools", 0, 10, 5)
        nearby_hospitals = st.slider("Nearby Hospitals", 0, 10, 5)
        parking_spaces = st.slider("Parking Spaces", 0, 5, 2)
        
    with col2:
        st.subheader("Prediction Results")
        
        # Calculate price per sq ft
        price_per_sqft = price_lakhs * 100000 / size_sqft
        
        # Simple prediction logic (without trained model)
        # Classification: Good Investment
        city_median = df[df['City'] == city]['Price_in_Lakhs'].median()
        is_good_investment = price_lakhs <= city_median
        
        # Regression: Future Price (8% annual growth for 5 years)
        future_price = price_lakhs * (1.08 ** 5)
        
        # Display results
        st.markdown("---")
        
        # Investment recommendation
        if is_good_investment:
            st.markdown('<div class="good-investment">✅ GOOD INVESTMENT</div>', unsafe_allow_html=True)
            st.success(f"This property is priced at or below the median price for {city} (₹{city_median:.2f}L)")
        else:
            st.markdown('<div class="bad-investment">❌ NOT RECOMMENDED</div>', unsafe_allow_html=True)
            st.warning(f"This property is priced above the median price for {city} (₹{city_median:.2f}L)")
        
        st.markdown("---")
        
        # Future price prediction
        st.metric(
            label="Estimated Price in 5 Years",
            value=f"₹{future_price:.2f} Lakhs",
            delta=f"+₹{future_price - price_lakhs:.2f}L ({((future_price/price_lakhs - 1)*100):.1f}%)"
        )
        
        # Additional metrics
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Price per Sq Ft", f"₹{price_per_sqft:.2f}")
        with col_b:
            st.metric("City Median Price", f"₹{city_median:.2f}L")
        
        # Investment insights
        st.markdown("---")
        st.subheader("Investment Insights")
        
        roi = ((future_price - price_lakhs) / price_lakhs) * 100
        annual_roi = roi / 5
        
        st.info(f"""
        **Return on Investment (ROI):**
        - 5-Year ROI: {roi:.2f}%
        - Annual ROI: {annual_roi:.2f}%
        - Total Gain: ₹{future_price - price_lakhs:.2f} Lakhs
        """)

def analytics_page(df):
    """Analytics dashboard page"""
    st.markdown('<div class="sub-header">Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Filters
    st.sidebar.subheader("Filters")
    
    # State filter
    states = ['All'] + sorted(df['State'].unique().tolist())
    selected_state = st.sidebar.selectbox("Filter by State", states)
    
    # BHK filter
    bhk_options = ['All'] + sorted(df['BHK'].unique().tolist())
    selected_bhk = st.sidebar.selectbox("Filter by BHK", bhk_options)
    
    # Price range filter
    price_range = st.sidebar.slider(
        "Price Range (Lakhs)",
        float(df['Price_in_Lakhs'].min()),
        float(df['Price_in_Lakhs'].max()),
        (float(df['Price_in_Lakhs'].min()), float(df['Price_in_Lakhs'].max()))
    )
    
    # Apply filters
    filtered_df = df.copy()
    if selected_state != 'All':
        filtered_df = filtered_df[filtered_df['State'] == selected_state]
    if selected_bhk != 'All':
        filtered_df = filtered_df[filtered_df['BHK'] == selected_bhk]
    filtered_df = filtered_df[
        (filtered_df['Price_in_Lakhs'] >= price_range[0]) &
        (filtered_df['Price_in_Lakhs'] <= price_range[1])
    ]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Properties", f"{len(filtered_df):,}")
    with col2:
        st.metric("Avg Price", f"₹{filtered_df['Price_in_Lakhs'].mean():.2f}L")
    with col3:
        st.metric("Avg Size", f"{filtered_df['Size_in_SqFt'].mean():.0f} sq ft")
    with col4:
        good_inv_pct = (filtered_df['Good_Investment'].sum() / len(filtered_df)) * 100
        st.metric("Good Investments", f"{good_inv_pct:.1f}%")
    
    # Visualizations
    st.markdown("---")
    
    # Row 1: Price distribution and BHK distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Distribution")
        fig1 = px.histogram(filtered_df, x='Price_in_Lakhs', nbins=50,
                           title="Distribution of Property Prices",
                           labels={'Price_in_Lakhs': 'Price (Lakhs)'})
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("BHK Distribution")
        bhk_counts = filtered_df['BHK'].value_counts().sort_index()
        fig2 = px.bar(x=bhk_counts.index, y=bhk_counts.values,
                     title="Properties by BHK",
                     labels={'x': 'BHK', 'y': 'Count'})
        st.plotly_chart(fig2, use_container_width=True)
    
    # Row 2: City-wise analysis
    st.markdown("---")
    st.subheader("City-wise Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        city_avg_price = filtered_df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(10)
        fig3 = px.bar(x=city_avg_price.values, y=city_avg_price.index,
                     orientation='h',
                     title="Top 10 Cities by Average Price",
                     labels={'x': 'Average Price (Lakhs)', 'y': 'City'})
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        city_counts = filtered_df['City'].value_counts().head(10)
        fig4 = px.bar(x=city_counts.index, y=city_counts.values,
                     title="Top 10 Cities by Property Count",
                     labels={'x': 'City', 'y': 'Count'})
        fig4.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig4, use_container_width=True)
    
    # Row 3: Size vs Price scatter
    st.markdown("---")
    st.subheader("Size vs Price Relationship")
    
    fig5 = px.scatter(filtered_df.sample(min(5000, len(filtered_df))), 
                     x='Size_in_SqFt', y='Price_in_Lakhs',
                     color='BHK',
                     title="Property Size vs Price",
                     labels={'Size_in_SqFt': 'Size (Sq Ft)', 'Price_in_Lakhs': 'Price (Lakhs)'},
                     opacity=0.6)
    st.plotly_chart(fig5, use_container_width=True)

def market_insights_page(df):
    """Market insights page"""
    st.markdown('<div class="sub-header">Market Insights</div>', unsafe_allow_html=True)
    
    # Property Type Analysis
    st.subheader("Property Type Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        type_avg_price = df.groupby('Property_Type')['Price_in_Lakhs'].mean().sort_values(ascending=False)
        fig1 = px.bar(x=type_avg_price.index, y=type_avg_price.values,
                     title="Average Price by Property Type",
                     labels={'x': 'Property Type', 'y': 'Average Price (Lakhs)'})
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        type_counts = df['Property_Type'].value_counts()
        fig2 = px.pie(values=type_counts.values, names=type_counts.index,
                     title="Property Type Distribution")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Furnished Status Analysis
    st.markdown("---")
    st.subheader("Furnished Status Impact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        furnished_avg = df.groupby('Furnished_Status')['Price_in_Lakhs'].mean().sort_values(ascending=False)
        fig3 = px.bar(x=furnished_avg.index, y=furnished_avg.values,
                     title="Average Price by Furnished Status",
                     labels={'x': 'Furnished Status', 'y': 'Average Price (Lakhs)'})
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        furnished_counts = df['Furnished_Status'].value_counts()
        fig4 = px.pie(values=furnished_counts.values, names=furnished_counts.index,
                     title="Furnished Status Distribution")
        st.plotly_chart(fig4, use_container_width=True)
    
    # Investment Opportunities
    st.markdown("---")
    st.subheader("Investment Opportunities")
    
    # Good investments by city
    good_inv_by_city = df[df['Good_Investment'] == 1].groupby('City').size().sort_values(ascending=False).head(10)
    
    fig5 = px.bar(x=good_inv_by_city.index, y=good_inv_by_city.values,
                 title="Top 10 Cities with Most Good Investment Opportunities",
                 labels={'x': 'City', 'y': 'Number of Good Investments'})
    fig5.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig5, use_container_width=True)
    
    # Key insights
    st.markdown("---")
    st.subheader("Key Market Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        most_expensive_city = df.groupby('City')['Price_in_Lakhs'].mean().idxmax()
        avg_price_expensive = df.groupby('City')['Price_in_Lakhs'].mean().max()
        st.info(f"""
        **Most Expensive City:**
        
        {most_expensive_city}
        
        Avg Price: ₹{avg_price_expensive:.2f}L
        """)
    
    with col2:
        most_affordable_city = df.groupby('City')['Price_in_Lakhs'].mean().idxmin()
        avg_price_affordable = df.groupby('City')['Price_in_Lakhs'].mean().min()
        st.success(f"""
        **Most Affordable City:**
        
        {most_affordable_city}
        
        Avg Price: ₹{avg_price_affordable:.2f}L
        """)
    
    with col3:
        best_roi_city = df.groupby('City')['Future_Price_5Y'].mean().idxmax()
        avg_future_price = df.groupby('City')['Future_Price_5Y'].mean().max()
        st.warning(f"""
        **Best ROI Potential:**
        
        {best_roi_city}
        
        Avg Future Price: ₹{avg_future_price:.2f}L
        """)

def eda_results_page(df):
    """EDA Results page showing all 20 questions and visualizations"""
    st.markdown('<div class="sub-header">📉 Exploratory Data Analysis Results</div>', unsafe_allow_html=True)
    
    st.markdown("""
    This page presents the comprehensive Exploratory Data Analysis (EDA) conducted on 250,000 properties.
    All 20 research questions have been answered with detailed visualizations.
    """)
    
    # Check if EDA images exist
    eda_path = "data/processed/"
    
    # Create tabs for different categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Price & Size (Q1-5)", 
        "🗺️ Location (Q6-10)", 
        "🔗 Correlations (Q11-15)", 
        "💰 Investment (Q16-20)"
    ])
    
    # Tab 1: Price & Size Analysis
    with tab1:
        st.header("Price & Size Analysis")
        
        # Q1: Price Distribution
        st.subheader("Q1: What is the distribution of property prices?")
        try:
            st.image(f"{eda_path}q1_price_distribution.png", use_container_width=True)
            st.markdown(f"""
            **Key Findings:**
            - Mean Price: ₹{df['Price_in_Lakhs'].mean():.2f} Lakhs
            - Median Price: ₹{df['Price_in_Lakhs'].median():.2f} Lakhs
            - Std Dev: ₹{df['Price_in_Lakhs'].std():.2f} Lakhs
            """)
        except:
            st.warning("Visualization not found. Please run: `python src/eda_analysis.py`")
        
        st.markdown("---")
        
        # Q2: Size Distribution
        st.subheader("Q2: What is the distribution of property sizes?")
        try:
            st.image(f"{eda_path}q2_size_distribution.png", use_container_width=True)
            st.markdown(f"""
            **Key Findings:**
            - Mean Size: {df['Size_in_SqFt'].mean():.0f} sq ft
            - Median Size: {df['Size_in_SqFt'].median():.0f} sq ft
            - Range: {df['Size_in_SqFt'].min():.0f} - {df['Size_in_SqFt'].max():.0f} sq ft
            """)
        except:
            st.warning("Visualization not found.")
        
        st.markdown("---")
        
        # Q3: Price per Sq Ft by Property Type
        st.subheader("Q3: How does price per sq ft vary by property type?")
        try:
            st.image(f"{eda_path}q3_price_per_sqft_by_type.png", use_container_width=True)
            avg_by_type = df.groupby('Property_Type')['Price_per_SqFt'].mean().sort_values(ascending=False)
            st.markdown("**Average Price per Sq Ft by Type:**")
            for ptype, price in avg_by_type.items():
                st.write(f"- {ptype}: ₹{price:.2f}/sq ft")
        except:
            st.warning("Visualization not found.")
        
        st.markdown("---")
        
        # Q4: Size vs Price Relationship
        st.subheader("Q4: Is there a relationship between property size and price?")
        try:
            st.image(f"{eda_path}q4_size_vs_price.png", use_container_width=True)
            correlation = df['Size_in_SqFt'].corr(df['Price_in_Lakhs'])
            st.markdown(f"""
            **Correlation Coefficient:** {correlation:.3f}
            
            {'Strong positive correlation - larger properties have higher prices' if correlation > 0.7 else 'Moderate positive correlation observed'}
            """)
        except:
            st.warning("Visualization not found.")
        
        st.markdown("---")
        
        # Q5: Outliers Analysis
        st.subheader("Q5: Are there outliers in price per sq ft or property size?")
        try:
            st.image(f"{eda_path}q5_outliers.png", use_container_width=True)
            Q1_price = df['Price_per_SqFt'].quantile(0.25)
            Q3_price = df['Price_per_SqFt'].quantile(0.75)
            IQR_price = Q3_price - Q1_price
            outliers_price = df[(df['Price_per_SqFt'] < Q1_price - 1.5*IQR_price) | (df['Price_per_SqFt'] > Q3_price + 1.5*IQR_price)]
            st.markdown(f"""
            **Outliers Detected:**
            - Price per Sq Ft: {len(outliers_price):,} outliers ({len(outliers_price)/len(df)*100:.2f}%)
            """)
        except:
            st.warning("Visualization not found.")
    
    # Tab 2: Location-Based Analysis
    with tab2:
        st.header("Location-Based Analysis")
        
        # Q6: Price by State
        st.subheader("Q6: What is the average price per sq ft by state?")
        try:
            st.image(f"{eda_path}q6_price_by_state.png", use_container_width=True)
            top_states = df.groupby('State')['Price_per_SqFt'].mean().sort_values(ascending=False).head(5)
            st.markdown("**Top 5 Most Expensive States:**")
            for state, price in top_states.items():
                st.write(f"- {state}: ₹{price:.2f}/sq ft")
        except:
            st.warning("Visualization not found.")
        
        st.markdown("---")
        
        # Q7: Price by City
        st.subheader("Q7: What is the average property price by city?")
        try:
            st.image(f"{eda_path}q7_price_by_city.png", use_container_width=True)
            top_cities = df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(5)
            st.markdown("**Top 5 Most Expensive Cities:**")
            for city, price in top_cities.items():
                st.write(f"- {city}: ₹{price:.2f} Lakhs")
        except:
            st.warning("Visualization not found.")
        
        st.markdown("---")
        
        # Q8: Age by Locality
        st.subheader("Q8: What is the median age of properties by locality?")
        try:
            st.image(f"{eda_path}q8_age_by_locality.png", use_container_width=True)
            st.markdown("Analysis shows property age distribution across different localities.")
        except:
            st.warning("Visualization not found.")
        
        st.markdown("---")
        
        # Q9: BHK Distribution by City
        st.subheader("Q9: How is BHK distributed across cities?")
        try:
            st.image(f"{eda_path}q9_bhk_by_city.png", use_container_width=True)
            st.markdown("Stacked bar chart showing BHK preferences in top 10 cities.")
        except:
            st.warning("Visualization not found.")
        
        st.markdown("---")
        
        # Q10: Top Expensive Localities
        st.subheader("Q10: What are the price trends for the top 5 most expensive localities?")
        try:
            st.image(f"{eda_path}q10_top_expensive_localities.png", use_container_width=True)
            top_localities = df.groupby('Locality')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(5)
            st.markdown("**Top 5 Premium Localities:**")
            for locality, price in top_localities.items():
                st.write(f"- {locality}: ₹{price:.2f} Lakhs")
        except:
            st.warning("Visualization not found.")
    
    # Tab 3: Feature Relationships & Correlations
    with tab3:
        st.header("Feature Relationships & Correlations")
        
        # Q11: Correlation Matrix
        st.subheader("Q11: How are numeric features correlated with each other?")
        try:
            st.image(f"{eda_path}q11_correlation_matrix.png", use_container_width=True)
            st.markdown("Comprehensive correlation heatmap showing relationships between all numeric features.")
        except:
            st.warning("Visualization not found.")
        
        st.markdown("---")
        
        # Q12: Schools vs Price
        st.subheader("Q12: How do nearby schools relate to price per sq ft?")
        try:
            st.image(f"{eda_path}q12_schools_vs_price.png", use_container_width=True)
            correlation = df['Nearby_Schools'].corr(df['Price_per_SqFt'])
            st.markdown(f"**Correlation:** {correlation:.3f} - {'Positive' if correlation > 0 else 'Negative'} relationship observed")
        except:
            st.warning("Visualization not found.")
        
        st.markdown("---")
        
        # Q13: Hospitals vs Price
        st.subheader("Q13: How do nearby hospitals relate to price per sq ft?")
        try:
            st.image(f"{eda_path}q13_hospitals_vs_price.png", use_container_width=True)
            correlation = df['Nearby_Hospitals'].corr(df['Price_per_SqFt'])
            st.markdown(f"**Correlation:** {correlation:.3f} - Healthcare accessibility impact measured")
        except:
            st.warning("Visualization not found.")
        
        st.markdown("---")
        
        # Q14: Price by Furnished Status
        st.subheader("Q14: How does price vary by furnished status?")
        try:
            st.image(f"{eda_path}q14_price_by_furnished.png", use_container_width=True)
            furnished_avg = df.groupby('Furnished_Status')['Price_in_Lakhs'].mean().sort_values(ascending=False)
            st.markdown("**Average Price by Furnished Status:**")
            for status, price in furnished_avg.items():
                st.write(f"- {status}: ₹{price:.2f} Lakhs")
        except:
            st.warning("Visualization not found.")
        
        st.markdown("---")
        
        # Q15: Price by Facing Direction
        st.subheader("Q15: How does price per sq ft vary by facing direction?")
        try:
            st.image(f"{eda_path}q15_price_by_facing.png", use_container_width=True)
            facing_avg = df.groupby('Facing')['Price_per_SqFt'].mean().sort_values(ascending=False)
            st.markdown("**Average Price per Sq Ft by Facing:**")
            for facing, price in facing_avg.head(5).items():
                st.write(f"- {facing}: ₹{price:.2f}/sq ft")
        except:
            st.warning("Visualization not found.")
    
    # Tab 4: Investment/Amenities/Ownership Analysis
    with tab4:
        st.header("Investment, Amenities & Ownership Analysis")
        
        # Q16: Owner Type Distribution
        st.subheader("Q16: How many properties belong to each owner type?")
        try:
            st.image(f"{eda_path}q16_owner_type.png", use_container_width=True)
            owner_counts = df['Owner_Type'].value_counts()
            st.markdown("**Distribution:**")
            for owner, count in owner_counts.items():
                st.write(f"- {owner}: {count:,} properties ({count/len(df)*100:.1f}%)")
        except:
            st.warning("Visualization not found.")
        
        st.markdown("---")
        
        # Q17: Availability Status
        st.subheader("Q17: How many properties are available under each availability status?")
        try:
            st.image(f"{eda_path}q17_availability_status.png", use_container_width=True)
            status_counts = df['Availability_Status'].value_counts()
            st.markdown("**Availability Distribution:**")
            for status, count in status_counts.items():
                st.write(f"- {status}: {count:,} properties")
        except:
            st.warning("Visualization not found.")
        
        st.markdown("---")
        
        # Q18: Parking vs Price
        st.subheader("Q18: Does parking space affect property price?")
        try:
            st.image(f"{eda_path}q18_parking_vs_price.png", use_container_width=True)
            parking_avg = df.groupby('Parking_Space')['Price_in_Lakhs'].mean().sort_index()
            st.markdown("**Average Price by Parking Spaces:**")
            for spaces, price in parking_avg.items():
                st.write(f"- {spaces} space(s): ₹{price:.2f} Lakhs")
        except:
            st.warning("Visualization not found.")
        
        st.markdown("---")
        
        # Q19: Amenities vs Price
        st.subheader("Q19: How do amenities affect price per sq ft?")
        try:
            st.image(f"{eda_path}q19_amenities_vs_price.png", use_container_width=True)
            st.markdown(f"Analysis of {df['Amenities'].nunique()} unique amenity combinations and their impact on pricing.")
        except:
            st.warning("Visualization not found.")
        
        st.markdown("---")
        
        # Q20: Transport Accessibility vs Price
        st.subheader("Q20: How does public transport accessibility relate to price per sq ft?")
        try:
            st.image(f"{eda_path}q20_transport_vs_price.png", use_container_width=True)
            transport_avg = df.groupby('Public_Transport_Accessibility')['Price_per_SqFt'].mean().sort_values(ascending=False)
            st.markdown("**Average Price per Sq Ft by Transport Accessibility:**")
            for level, price in transport_avg.items():
                st.write(f"- {level}: ₹{price:.4f}/sq ft")
        except:
            st.warning("Visualization not found.")
    
    # Summary section
    st.markdown("---")
    st.success(f"""
    ### 📊 EDA Summary
    
    **Dataset Statistics:**
    - Total Properties Analyzed: {len(df):,}
    - States Covered: {df['State'].nunique()}
    - Cities Covered: {df['City'].nunique()}
    - Localities Covered: {df['Locality'].nunique()}
    
    **Key Insights:**
    - All 20 EDA questions have been comprehensively answered
    - Visualizations saved in: `data/processed/`
    - Detailed analysis available in: `EDA_Summary.md`
    
    **To regenerate visualizations, run:**
    ```
    python src/eda_analysis.py
    ```
    """)

def about_page():
    """About page with project and creator information"""
    st.markdown('<div class="sub-header">About This Application</div>', unsafe_allow_html=True)
    
    # Project Overview
    st.markdown("""
    ## 🏠 Real Estate Investment Advisor
    
    A comprehensive machine learning application that analyzes 250,000 properties across India to predict 
    investment viability and future property values using advanced classification and regression models.
    
    ### 🎯 Key Features
    
    1. **🔮 Investment Prediction**
       - Binary classification: "Good Investment" vs "Not Recommended"
       - Based on comprehensive market analysis and city median prices
       - ROI calculations and investment insights
    
    2. **📈 Future Price Estimation**
       - Predicts property value after 5 years
       - Uses regression models with 8% annual growth baseline
       - Provides detailed price projections
    
    3. **📊 Interactive Analytics Dashboard**
       - Real-time data visualizations
       - City-wise, BHK-wise, and property type analysis
       - Dynamic filters for custom exploration
    
    4. **🗺️ Market Insights**
       - Identifies investment opportunities by location
       - Compares cities and property types
       - Provides key market statistics and trends
    
    5. **📉 Comprehensive EDA Results**
       - All 20 research questions answered
       - Interactive visualizations organized by category
       - Statistical summaries and insights
    
    ### 📊 Dataset Information
    
    - **Size:** 250,000 properties across India
    - **Original Features:** 23 property attributes
    - **Engineered Features:** 13 additional features
    - **Geographic Coverage:** 20 states, 42 cities, 500 localities
    - **Target Variables:** 
      - Good_Investment (Classification)
      - Future_Price_5Y (Regression)
    
    ### 🔧 Technology Stack
    
    **Frontend & Visualization:**
    - Streamlit (Web Application)
    - Plotly (Interactive Charts)
    - Matplotlib & Seaborn (Static Visualizations)
    
    **Data Processing:**
    - Pandas (Data Manipulation)
    - NumPy (Numerical Computing)
    
    **Machine Learning:**
    - Scikit-learn (Traditional ML Models)
    - XGBoost (Gradient Boosting)
    
    **ML Operations:**
    - MLflow (Experiment Tracking & Model Registry)
    
    ### 🤖 Machine Learning Models
    
    **Classification Models (7):**
    - Logistic Regression
    - Random Forest Classifier
    - XGBoost Classifier
    - Decision Tree Classifier
    - Gradient Boosting Classifier
    - Naive Bayes
    - AdaBoost Classifier
    
    **Regression Models (6):**
    - Linear Regression
    - Random Forest Regressor
    - XGBoost Regressor
    - Decision Tree Regressor
    - Gradient Boosting Regressor
    - Ridge Regression
    
    ### 📈 Project Highlights
    
    - ✅ **20 EDA Questions** answered with visualizations
    - ✅ **13 ML Models** trained and compared
    - ✅ **MLflow Integration** for experiment tracking
    - ✅ **5-Page Web Application** with interactive features
    - ✅ **Comprehensive Documentation** and code quality
    
    ---
    """)
    
    # Creator Information
    st.markdown('<div class="sub-header">👨‍💻 About the Creator</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=150)
        st.markdown("### K. Natrajan")
        st.caption("Lead Automation Engineer")
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/)")
    
    with col2:
        st.info("""
        **With over 7.6 years of experience in the software industry**, I specialize in building robust 
        automation frameworks that ensure data integrity and system reliability. My career has been defined 
        by a transition from manual validation to architecting sophisticated automation pipelines for global 
        leaders like Mindtree and RNTBCI.
        """)
        
        st.markdown("#### 🚀 Industry Expertise")
        st.write("""
        I have a proven track record in Banking and Supply Chain Finance, developing solutions that automate 
        end-to-end office software for wholesale petroleum and petroleum-related financial transactions.
        """)
        
        st.markdown("#### 💻 Technical Core")
        st.write("""
        My expertise lies in **Selenium with Java**, **API Automation (Rest Assured)**, and **SQL-based data validation**. 
        Now expanding into **Data Science and Machine Learning** through this capstone project.
        """)
        
        st.markdown("#### 🏆 Strategic Leadership")
        st.write("""
        Beyond writing scripts, I lead QA teams, mentor junior engineers, and manage CI/CD execution via Jenkins 
        to ensure **95% automation coverage** for production releases.
        """)
        
        st.markdown("#### 🎓 Education")
        st.write("""
        I hold a **Master of Computer Applications (MCA)**, reinforcing my practical experience with deep 
        theoretical knowledge.
        """)
    
    st.markdown("---")
    
    # Project Information
    st.markdown("""
    ### 📚 Project Context
    
    This **Real Estate Investment Advisor** is a capstone project for the **GUVI Data Science Course**, 
    demonstrating:
    
    - ✅ End-to-end machine learning pipeline development
    - ✅ Comprehensive exploratory data analysis (20 questions)
    - ✅ Multiple model training and comparison (13 models)
    - ✅ MLflow experiment tracking and model management
    - ✅ Interactive web application deployment with Streamlit
    - ✅ Professional documentation and code quality
    
    ### 🎯 Learning Outcomes
    
    Through this project, I have demonstrated proficiency in:
    - Data preprocessing and feature engineering
    - Statistical analysis and visualization
    - Classification and regression modeling
    - Model evaluation and comparison
    - ML operations and experiment tracking
    - Web application development
    - Project documentation and presentation
    
    ---
    
    ### 📞 Contact & Links
    
    - **Course:** GUVI Data Science Capstone Project
    - **Submission Date:** February 2026
    - **GitHub Repository:** [git@github.com:natrajan0409/GUVI-_project-AIML_Real-Estate-Investment-Advisor.git]
    
    ---
    
    **Note:** This application currently uses rule-based prediction logic for demonstration. 
    The trained ML models from MLflow can be integrated for production-grade predictions.
    
    **⭐ Thank you for exploring this project!**
    """)

if __name__ == "__main__":
    main()
