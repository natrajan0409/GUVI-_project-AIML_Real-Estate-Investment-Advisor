"""
Exploratory Data Analysis Module for Real Estate Investment Advisor
Answers all 20 EDA questions with comprehensive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class RealEstateEDA:
    def __init__(self, data_path):
        """Initialize EDA with processed data"""
        self.df = pd.read_csv(data_path)
        print(f"Data loaded for EDA: {self.df.shape}")
        
    # ========== PRICE & SIZE ANALYSIS (Questions 1-5) ==========
    
    def q1_price_distribution(self):
        """Q1: What is the distribution of property prices?"""
        print("\n" + "="*80)
        print("Q1: Distribution of Property Prices")
        print("="*80)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        axes[0].hist(self.df['Price_in_Lakhs'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0].set_xlabel('Price (in Lakhs)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Property Prices')
        axes[0].axvline(self.df['Price_in_Lakhs'].median(), color='red', linestyle='--', label=f'Median: Rs.{self.df["Price_in_Lakhs"].median():.2f}L')
        axes[0].legend()
        
        # Box plot
        axes[1].boxplot(self.df['Price_in_Lakhs'], vert=True)
        axes[1].set_ylabel('Price (in Lakhs)')
        axes[1].set_title('Box Plot of Property Prices')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/processed/q1_price_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Mean Price: Rs.{self.df['Price_in_Lakhs'].mean():.2f} Lakhs")
        print(f"Median Price: Rs.{self.df['Price_in_Lakhs'].median():.2f} Lakhs")
        print(f"Std Dev: Rs.{self.df['Price_in_Lakhs'].std():.2f} Lakhs")
        
    def q2_size_distribution(self):
        """Q2: What is the distribution of property sizes?"""
        print("\n" + "="*80)
        print("Q2: Distribution of Property Sizes")
        print("="*80)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        axes[0].hist(self.df['Size_in_SqFt'], bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
        axes[0].set_xlabel('Size (in Sq Ft)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Property Sizes')
        axes[0].axvline(self.df['Size_in_SqFt'].median(), color='red', linestyle='--', label=f'Median: {self.df["Size_in_SqFt"].median():.0f} sq ft')
        axes[0].legend()
        
        # Box plot
        axes[1].boxplot(self.df['Size_in_SqFt'], vert=True)
        axes[1].set_ylabel('Size (in Sq Ft)')
        axes[1].set_title('Box Plot of Property Sizes')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/processed/q2_size_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Mean Size: {self.df['Size_in_SqFt'].mean():.2f} sq ft")
        print(f"Median Size: {self.df['Size_in_SqFt'].median():.2f} sq ft")
        
    def q3_price_per_sqft_by_property_type(self):
        """Q3: How does the price per sq ft vary by property type?"""
        print("\n" + "="*80)
        print("Q3: Price per Sq Ft by Property Type")
        print("="*80)
        
        if 'Property_Type' in self.df.columns:
            avg_price_per_sqft = self.df.groupby('Property_Type')['Price_per_SqFt'].mean().sort_values(ascending=False)
            
            plt.figure(figsize=(12, 6))
            avg_price_per_sqft.plot(kind='bar', color='coral', edgecolor='black')
            plt.xlabel('Property Type')
            plt.ylabel('Average Price per Sq Ft (Rs.)')
            plt.title('Average Price per Sq Ft by Property Type')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('data/processed/q3_price_per_sqft_by_type.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(avg_price_per_sqft)
        
    def q4_size_vs_price_relationship(self):
        """Q4: Is there a relationship between property size and price?"""
        print("\n" + "="*80)
        print("Q4: Relationship between Property Size and Price")
        print("="*80)
        
        correlation = self.df['Size_in_SqFt'].corr(self.df['Price_in_Lakhs'])
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['Size_in_SqFt'], self.df['Price_in_Lakhs'], alpha=0.5, s=10)
        plt.xlabel('Size (in Sq Ft)')
        plt.ylabel('Price (in Lakhs)')
        plt.title(f'Size vs Price (Correlation: {correlation:.3f})')
        
        # Add trend line
        z = np.polyfit(self.df['Size_in_SqFt'], self.df['Price_in_Lakhs'], 1)
        p = np.poly1d(z)
        plt.plot(self.df['Size_in_SqFt'], p(self.df['Size_in_SqFt']), "r--", alpha=0.8, linewidth=2)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('data/processed/q4_size_vs_price.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Correlation coefficient: {correlation:.3f}")
        
    def q5_outliers_analysis(self):
        """Q5: Are there any outliers in price per sq ft or property size?"""
        print("\n" + "="*80)
        print("Q5: Outliers in Price per Sq Ft and Property Size")
        print("="*80)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Price per Sq Ft outliers
        axes[0].boxplot(self.df['Price_per_SqFt'], vert=True)
        axes[0].set_ylabel('Price per Sq Ft (Rs.)')
        axes[0].set_title('Outliers in Price per Sq Ft')
        axes[0].grid(True, alpha=0.3)
        
        # Property Size outliers
        axes[1].boxplot(self.df['Size_in_SqFt'], vert=True)
        axes[1].set_ylabel('Size (in Sq Ft)')
        axes[1].set_title('Outliers in Property Size')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/processed/q5_outliers.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate outliers
        for col in ['Price_per_SqFt', 'Size_in_SqFt']:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[(self.df[col] < Q1 - 1.5*IQR) | (self.df[col] > Q3 + 1.5*IQR)]
            print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(self.df)*100:.2f}%)")
    
    # ========== LOCATION-BASED ANALYSIS (Questions 6-10) ==========
    
    def q6_avg_price_per_sqft_by_state(self):
        """Q6: What is the average price per sq ft by state?"""
        print("\n" + "="*80)
        print("Q6: Average Price per Sq Ft by State")
        print("="*80)
        
        if 'State' in self.df.columns:
            avg_by_state = self.df.groupby('State')['Price_per_SqFt'].mean().sort_values(ascending=False)
            
            plt.figure(figsize=(12, 6))
            avg_by_state.plot(kind='barh', color='steelblue', edgecolor='black')
            plt.xlabel('Average Price per Sq Ft (Rs.)')
            plt.ylabel('State')
            plt.title('Average Price per Sq Ft by State')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig('data/processed/q6_price_by_state.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(avg_by_state.head(10))
    
    def q7_avg_price_by_city(self):
        """Q7: What is the average property price by city?"""
        print("\n" + "="*80)
        print("Q7: Average Property Price by City")
        print("="*80)
        
        if 'City' in self.df.columns:
            avg_by_city = self.df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(20)
            
            plt.figure(figsize=(12, 8))
            avg_by_city.plot(kind='barh', color='teal', edgecolor='black')
            plt.xlabel('Average Price (in Lakhs)')
            plt.ylabel('City')
            plt.title('Top 20 Cities by Average Property Price')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig('data/processed/q7_price_by_city.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(avg_by_city.head(10))
    
    def q8_median_age_by_locality(self):
        """Q8: What is the median age of properties by locality?"""
        print("\n" + "="*80)
        print("Q8: Median Age of Properties by Locality")
        print("="*80)
        
        if 'Age_of_Property' in self.df.columns and 'Locality' in self.df.columns:
            median_age = self.df.groupby('Locality')['Age_of_Property'].median().sort_values(ascending=False).head(20)
            
            plt.figure(figsize=(12, 8))
            median_age.plot(kind='barh', color='orange', edgecolor='black')
            plt.xlabel('Median Age (Years)')
            plt.ylabel('Locality')
            plt.title('Top 20 Localities by Median Property Age')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig('data/processed/q8_age_by_locality.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(median_age.head(10))
    
    def q9_bhk_distribution_by_city(self):
        """Q9: How is BHK distributed across cities?"""
        print("\n" + "="*80)
        print("Q9: BHK Distribution Across Cities")
        print("="*80)
        
        if 'BHK' in self.df.columns and 'City' in self.df.columns:
            # Get top 10 cities by property count
            top_cities = self.df['City'].value_counts().head(10).index
            df_top_cities = self.df[self.df['City'].isin(top_cities)]
            
            bhk_city = pd.crosstab(df_top_cities['City'], df_top_cities['BHK'])
            
            bhk_city.plot(kind='bar', stacked=True, figsize=(14, 6), colormap='viridis')
            plt.xlabel('City')
            plt.ylabel('Count')
            plt.title('BHK Distribution Across Top 10 Cities')
            plt.legend(title='BHK', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('data/processed/q9_bhk_by_city.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(bhk_city)
    
    def q10_top_expensive_localities(self):
        """Q10: What are the price trends for the top 5 most expensive localities?"""
        print("\n" + "="*80)
        print("Q10: Price Trends for Top 5 Most Expensive Localities")
        print("="*80)
        
        if 'Locality' in self.df.columns:
            top_localities = self.df.groupby('Locality')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(5)
            
            plt.figure(figsize=(12, 6))
            top_localities.plot(kind='bar', color='crimson', edgecolor='black')
            plt.xlabel('Locality')
            plt.ylabel('Average Price (in Lakhs)')
            plt.title('Top 5 Most Expensive Localities')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('data/processed/q10_top_expensive_localities.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(top_localities)
    
    # ========== FEATURE RELATIONSHIP & CORRELATION (Questions 11-15) ==========
    
    def q11_correlation_matrix(self):
        """Q11: How are numeric features correlated with each other?"""
        print("\n" + "="*80)
        print("Q11: Correlation Matrix of Numeric Features")
        print("="*80)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(16, 12))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Numeric Features')
        plt.tight_layout()
        plt.savefig('data/processed/q11_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print top correlations with price
        price_corr = correlation_matrix['Price_in_Lakhs'].sort_values(ascending=False)
        print("\nTop correlations with Price:")
        print(price_corr.head(10))
    
    def q12_schools_vs_price(self):
        """Q12: How do nearby schools relate to price per sq ft?"""
        print("\n" + "="*80)
        print("Q12: Nearby Schools vs Price per Sq Ft")
        print("="*80)
        
        if 'Nearby_Schools' in self.df.columns:
            correlation = self.df['Nearby_Schools'].corr(self.df['Price_per_SqFt'])
            
            plt.figure(figsize=(10, 6))
            plt.scatter(self.df['Nearby_Schools'], self.df['Price_per_SqFt'], alpha=0.5, s=10)
            plt.xlabel('Number of Nearby Schools')
            plt.ylabel('Price per Sq Ft (Rs.)')
            plt.title(f'Nearby Schools vs Price per Sq Ft (Correlation: {correlation:.3f})')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('data/processed/q12_schools_vs_price.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Correlation: {correlation:.3f}")
    
    def q13_hospitals_vs_price(self):
        """Q13: How do nearby hospitals relate to price per sq ft?"""
        print("\n" + "="*80)
        print("Q13: Nearby Hospitals vs Price per Sq Ft")
        print("="*80)
        
        if 'Nearby_Hospitals' in self.df.columns:
            correlation = self.df['Nearby_Hospitals'].corr(self.df['Price_per_SqFt'])
            
            plt.figure(figsize=(10, 6))
            plt.scatter(self.df['Nearby_Hospitals'], self.df['Price_per_SqFt'], alpha=0.5, s=10, color='green')
            plt.xlabel('Number of Nearby Hospitals')
            plt.ylabel('Price per Sq Ft (Rs.)')
            plt.title(f'Nearby Hospitals vs Price per Sq Ft (Correlation: {correlation:.3f})')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('data/processed/q13_hospitals_vs_price.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Correlation: {correlation:.3f}")
    
    def q14_price_by_furnished_status(self):
        """Q14: How does price vary by furnished status?"""
        print("\n" + "="*80)
        print("Q14: Price by Furnished Status")
        print("="*80)
        
        if 'Furnished_Status' in self.df.columns:
            avg_price = self.df.groupby('Furnished_Status')['Price_in_Lakhs'].mean().sort_values(ascending=False)
            
            plt.figure(figsize=(10, 6))
            avg_price.plot(kind='bar', color='purple', edgecolor='black')
            plt.xlabel('Furnished Status')
            plt.ylabel('Average Price (in Lakhs)')
            plt.title('Average Price by Furnished Status')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('data/processed/q14_price_by_furnished.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(avg_price)
    
    def q15_price_per_sqft_by_facing(self):
        """Q15: How does price per sq ft vary by property facing direction?"""
        print("\n" + "="*80)
        print("Q15: Price per Sq Ft by Facing Direction")
        print("="*80)
        
        if 'Facing' in self.df.columns:
            avg_price = self.df.groupby('Facing')['Price_per_SqFt'].mean().sort_values(ascending=False)
            
            plt.figure(figsize=(10, 6))
            avg_price.plot(kind='bar', color='gold', edgecolor='black')
            plt.xlabel('Facing Direction')
            plt.ylabel('Average Price per Sq Ft (Rs.)')
            plt.title('Average Price per Sq Ft by Facing Direction')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('data/processed/q15_price_by_facing.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(avg_price)
    
    # ========== INVESTMENT/AMENITIES/OWNERSHIP ANALYSIS (Questions 16-20) ==========
    
    def q16_owner_type_distribution(self):
        """Q16: How many properties belong to each owner type?"""
        print("\n" + "="*80)
        print("Q16: Owner Type Distribution")
        print("="*80)
        
        if 'Owner_Type' in self.df.columns:
            owner_counts = self.df['Owner_Type'].value_counts()
            
            plt.figure(figsize=(10, 6))
            owner_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
            plt.ylabel('')
            plt.title('Distribution of Properties by Owner Type')
            plt.tight_layout()
            plt.savefig('data/processed/q16_owner_type.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(owner_counts)
    
    def q17_availability_status_distribution(self):
        """Q17: How many properties are available under each availability status?"""
        print("\n" + "="*80)
        print("Q17: Availability Status Distribution")
        print("="*80)
        
        if 'Availability_Status' in self.df.columns:
            status_counts = self.df['Availability_Status'].value_counts()
            
            plt.figure(figsize=(10, 6))
            status_counts.plot(kind='bar', color='lightcoral', edgecolor='black')
            plt.xlabel('Availability Status')
            plt.ylabel('Count')
            plt.title('Distribution of Properties by Availability Status')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('data/processed/q17_availability_status.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(status_counts)
    
    def q18_parking_vs_price(self):
        """Q18: Does parking space affect property price?"""
        print("\n" + "="*80)
        print("Q18: Parking Space vs Property Price")
        print("="*80)
        
        if 'Parking_Space' in self.df.columns:
            avg_price = self.df.groupby('Parking_Space')['Price_in_Lakhs'].mean().sort_index()
            
            plt.figure(figsize=(10, 6))
            avg_price.plot(kind='bar', color='navy', edgecolor='black')
            plt.xlabel('Number of Parking Spaces')
            plt.ylabel('Average Price (in Lakhs)')
            plt.title('Average Price by Number of Parking Spaces')
            plt.xticks(rotation=0)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('data/processed/q18_parking_vs_price.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(avg_price)
    
    def q19_amenities_vs_price(self):
        """Q19: How do amenities affect price per sq ft?"""
        print("\n" + "="*80)
        print("Q19: Amenities vs Price per Sq Ft")
        print("="*80)
        
        if 'Amenities' in self.df.columns:
            avg_price = self.df.groupby('Amenities')['Price_per_SqFt'].mean().sort_values(ascending=False)
            
            plt.figure(figsize=(12, 6))
            avg_price.plot(kind='bar', color='mediumseagreen', edgecolor='black')
            plt.xlabel('Amenities')
            plt.ylabel('Average Price per Sq Ft (Rs.)')
            plt.title('Average Price per Sq Ft by Amenities')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('data/processed/q19_amenities_vs_price.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(avg_price)
    
    def q20_transport_vs_price(self):
        """Q20: How does public transport accessibility relate to price per sq ft?"""
        print("\n" + "="*80)
        print("Q20: Public Transport Accessibility vs Price per Sq Ft")
        print("="*80)
        
        if 'Public_Transport_Accessibility' in self.df.columns:
            avg_price = self.df.groupby('Public_Transport_Accessibility')['Price_per_SqFt'].mean().sort_values(ascending=False)
            
            plt.figure(figsize=(10, 6))
            avg_price.plot(kind='bar', color='darkblue', edgecolor='black')
            plt.xlabel('Public Transport Accessibility')
            plt.ylabel('Average Price per Sq Ft (Rs.)')
            plt.title('Average Price per Sq Ft by Public Transport Accessibility')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('data/processed/q20_transport_vs_price.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(avg_price)
    
    def run_all_eda(self):
        """Run all 20 EDA questions"""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE EDA - 20 QUESTIONS")
        print("="*80)
        
        # Price & Size Analysis (1-5)
        self.q1_price_distribution()
        self.q2_size_distribution()
        self.q3_price_per_sqft_by_property_type()
        self.q4_size_vs_price_relationship()
        self.q5_outliers_analysis()
        
        # Location-based Analysis (6-10)
        self.q6_avg_price_per_sqft_by_state()
        self.q7_avg_price_by_city()
        self.q8_median_age_by_locality()
        self.q9_bhk_distribution_by_city()
        self.q10_top_expensive_localities()
        
        # Feature Relationship & Correlation (11-15)
        self.q11_correlation_matrix()
        self.q12_schools_vs_price()
        self.q13_hospitals_vs_price()
        self.q14_price_by_furnished_status()
        self.q15_price_per_sqft_by_facing()
        
        # Investment/Amenities/Ownership Analysis (16-20)
        self.q16_owner_type_distribution()
        self.q17_availability_status_distribution()
        self.q18_parking_vs_price()
        self.q19_amenities_vs_price()
        self.q20_transport_vs_price()
        
        print("\n" + "="*80)
        print("[+] ALL 20 EDA QUESTIONS COMPLETED!")
        print("[+] Visualizations saved in data/processed/")
        print("="*80)


def main():
    """Main function to run EDA"""
    data_path = "data/processed/housing_data_processed.csv"
    
    eda = RealEstateEDA(data_path)
    eda.run_all_eda()
    
    print("\n[+] EDA Complete! Check data/processed/ for all visualizations.")


if __name__ == "__main__":
    main()
