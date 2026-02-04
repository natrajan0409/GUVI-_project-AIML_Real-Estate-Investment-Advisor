"""
Data Preprocessing Module for Real Estate Investment Advisor
Handles data loading, cleaning, feature engineering, and target variable creation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class RealEstateDataPreprocessor:
    def __init__(self, data_path):
        """Initialize the preprocessor with data path"""
        self.data_path = data_path
        self.df = None
        self.label_encoders = {}
        
    def load_dataset(self):
        """Load the raw dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully! Shape: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        return self.df
    
    def inspect_data(self):
        """Display basic information about the dataset"""
        print("\n" + "="*80)
        print("DATASET INSPECTION")
        print("="*80)
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"\nFirst few rows:")
        print(self.df.head())
        print(f"\nData Types:")
        print(self.df.dtypes)
        print(f"\nMissing Values:")
        print(self.df.isnull().sum())
        print(f"\nBasic Statistics:")
        print(self.df.describe())
        
    def handle_missing_values(self):
        """Handle missing values using appropriate strategies"""
        print("\n" + "="*80)
        print("HANDLING MISSING VALUES")
        print("="*80)
        
        missing_before = self.df.isnull().sum().sum()
        print(f"\nTotal missing values before: {missing_before}")
        
        # Numeric columns: fill with median
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                print(f"Filled {col} with median: {median_val}")
        
        # Categorical columns: fill with mode
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                mode_val = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown'
                self.df[col].fillna(mode_val, inplace=True)
                print(f"Filled {col} with mode: {mode_val}")
        
        missing_after = self.df.isnull().sum().sum()
        print(f"\nTotal missing values after: {missing_after}")
        
    def detect_outliers(self, columns=None):
        """Detect outliers using IQR method"""
        print("\n" + "="*80)
        print("OUTLIER DETECTION (IQR Method)")
        print("="*80)
        
        if columns is None:
            columns = ['Price_in_Lakhs', 'Size_in_SqFt', 'Price_per_SqFt']
        
        outlier_info = {}
        for col in columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                outlier_count = len(outliers)
                outlier_percentage = (outlier_count / len(self.df)) * 100
                
                outlier_info[col] = {
                    'count': outlier_count,
                    'percentage': outlier_percentage,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
                
                print(f"\n{col}:")
                print(f"  Outliers: {outlier_count} ({outlier_percentage:.2f}%)")
                print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return outlier_info
    
    def encode_categorical(self):
        """Encode categorical variables"""
        print("\n" + "="*80)
        print("ENCODING CATEGORICAL VARIABLES")
        print("="*80)
        
        categorical_cols = ['State', 'City', 'Locality', 'Property_Type', 
                           'Furnished_Status', 'Facing', 'Owner_Type', 
                           'Availability_Status', 'Security', 'Amenities']
        
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_Encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                print(f"Encoded {col}: {len(le.classes_)} unique values")
    
    def create_target_variables(self, growth_rate=0.08, years=5):
        """
        Create target variables for classification and regression
        
        Classification Target: Good_Investment
        - Based on price comparison with median price per city
        
        Regression Target: Future_Price_5Y
        - Calculate using compound growth rate
        """
        print("\n" + "="*80)
        print("CREATING TARGET VARIABLES")
        print("="*80)
        
        # Regression Target: Future Price (5 years)
        self.df['Future_Price_5Y'] = self.df['Price_in_Lakhs'] * ((1 + growth_rate) ** years)
        print(f"\n[+] Created 'Future_Price_5Y' using {growth_rate*100}% annual growth rate")
        print(f"  Current Price Range: Rs.{self.df['Price_in_Lakhs'].min():.2f}L - Rs.{self.df['Price_in_Lakhs'].max():.2f}L")
        print(f"  Future Price Range: Rs.{self.df['Future_Price_5Y'].min():.2f}L - Rs.{self.df['Future_Price_5Y'].max():.2f}L")
        
        # Classification Target: Good Investment
        # Strategy: Price <= median price per city
        if 'City' in self.df.columns:
            city_median_prices = self.df.groupby('City')['Price_in_Lakhs'].median()
            self.df['City_Median_Price'] = self.df['City'].map(city_median_prices)
            self.df['Good_Investment'] = (self.df['Price_in_Lakhs'] <= self.df['City_Median_Price']).astype(int)
        else:
            # Fallback: use overall median
            median_price = self.df['Price_in_Lakhs'].median()
            self.df['Good_Investment'] = (self.df['Price_in_Lakhs'] <= median_price).astype(int)
        
        good_investment_count = self.df['Good_Investment'].sum()
        good_investment_pct = (good_investment_count / len(self.df)) * 100
        
        print(f"\n[+] Created 'Good_Investment' (binary classification)")
        print(f"  Good Investments: {good_investment_count} ({good_investment_pct:.2f}%)")
        print(f"  Not Recommended: {len(self.df) - good_investment_count} ({100-good_investment_pct:.2f}%)")
        
    def save_processed_data(self, output_path):
        """Save the processed dataset"""
        print("\n" + "="*80)
        print("SAVING PROCESSED DATA")
        print("="*80)
        
        self.df.to_csv(output_path, index=False)
        print(f"\n[+] Processed data saved to: {output_path}")
        print(f"  Final shape: {self.df.shape}")
        
    def get_processed_data(self):
        """Return the processed dataframe"""
        return self.df
    
    def run_full_pipeline(self, output_path):
        """Run the complete preprocessing pipeline"""
        print("\n" + "="*80)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("="*80)
        
        self.load_dataset()
        self.inspect_data()
        self.handle_missing_values()
        self.detect_outliers()
        self.encode_categorical()
        self.create_target_variables()
        self.save_processed_data(output_path)
        
        print("\n" + "="*80)
        print("[SUCCESS] PREPROCESSING PIPELINE COMPLETED!")
        print("="*80)
        
        return self.df


def main():
    """Main function to run preprocessing"""
    # Paths
    input_path = "data/raw/india_housing_prices.csv"
    output_path = "data/processed/housing_data_processed.csv"
    
    # Initialize and run preprocessor
    preprocessor = RealEstateDataPreprocessor(input_path)
    df_processed = preprocessor.run_full_pipeline(output_path)
    
    print(f"\n[SUCCESS] Preprocessing complete! Processed data available at: {output_path}")


if __name__ == "__main__":
    main()
