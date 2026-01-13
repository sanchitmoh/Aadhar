#!/usr/bin/env python3
"""
Aadhaar API Integration Script
Demonstrates how to fetch data from the APIs mentioned in api.md
"""

import requests
import pandas as pd
import json
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

class AadhaarAPIClient:
    def __init__(self):
        self.api_key = os.getenv('AadhaarStatewise', '579b464db66ec23bdd000001a6537618e84d40aa5ae04945c503592f')
        self.base_urls = {
            'demographic': '/resource/19eac040-0b94-49fa-b239-4f2fd8677d53',
            'biometric': '/resource/65454dab-1517-40a3-ac1d-47d4dfe6891c',
            'enrollment': '/resource/65454dab-1517-40a3-ac1d-47d4dfe6891c'
        }
        
    def fetch_data(self, data_type, state=None, district=None, format='json', limit=100, offset=0):
        """
        Fetch data from Aadhaar API
        
        Parameters:
        - data_type: 'demographic', 'biometric', or 'enrollment'
        - state: Filter by state name
        - district: Filter by district name
        - format: 'json', 'xml', or 'csv'
        - limit: Maximum records to return
        - offset: Number of records to skip
        """
        
        if data_type not in self.base_urls:
            raise ValueError("data_type must be 'demographic', 'biometric', or 'enrollment'")
        
        # Build API URL (Note: You'll need to replace with actual API base URL)
        base_url = "https://api.data.gov.in"  # Replace with actual API base URL
        endpoint = base_url + self.base_urls[data_type]
        
        # Build parameters
        params = {
            'api-key': self.api_key,
            'format': format,
            'limit': limit,
            'offset': offset
        }
        
        if state:
            params['filters[state]'] = state
        if district:
            params['filters[district]'] = district
        
        try:
            print(f"🔄 Fetching {data_type} data...")
            print(f"   State: {state or 'All'}")
            print(f"   District: {district or 'All'}")
            print(f"   Format: {format}")
            print(f"   Limit: {limit}")
            
            response = requests.get(endpoint, params=params, timeout=30)
            
            if response.status_code == 200:
                print(f"✅ Successfully fetched {data_type} data")
                
                if format == 'json':
                    return response.json()
                elif format == 'csv':
                    return response.text
                else:
                    return response.text
                    
            else:
                print(f"❌ API request failed with status code: {response.status_code}")
                print(f"   Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
    
    def fetch_state_wise_data(self, data_type, states_list, format='json'):
        """
        Fetch data for multiple states
        """
        all_data = []
        
        for state in states_list:
            print(f"\n📍 Fetching data for {state}...")
            data = self.fetch_data(data_type, state=state, format=format, limit=1000)
            
            if data:
                all_data.append({
                    'state': state,
                    'data': data
                })
            
            # Add delay to avoid rate limiting
            time.sleep(1)
        
        return all_data
    
    def demonstrate_api_usage(self):
        """
        Demonstrate various API usage patterns
        """
        print("🚀 AADHAAR API INTEGRATION DEMONSTRATION")
        print("="*50)
        
        # List of states to test (from api.md)
        test_states = [
            'Tamil Nadu', 'Andhra Pradesh', 'Uttar Pradesh', 
            'West Bengal', 'Maharashtra', 'Karnataka'
        ]
        
        # 1. Fetch demographic data for specific states
        print("\n1. DEMOGRAPHIC DATA FETCHING")
        print("-" * 30)
        
        for state in test_states[:3]:  # Test first 3 states
            demo_data = self.fetch_data('demographic', state=state, limit=10)
            if demo_data:
                print(f"   ✅ {state}: Data fetched successfully")
            else:
                print(f"   ❌ {state}: Failed to fetch data")
        
        # 2. Fetch biometric data with different formats
        print("\n2. BIOMETRIC DATA FETCHING (Different Formats)")
        print("-" * 30)
        
        formats = ['json', 'csv', 'xml']
        for fmt in formats:
            bio_data = self.fetch_data('biometric', state='Tamil Nadu', 
                                     format=fmt, limit=5)
            if bio_data:
                print(f"   ✅ Format {fmt}: Success")
            else:
                print(f"   ❌ Format {fmt}: Failed")
        
        # 3. Pagination example
        print("\n3. PAGINATION DEMONSTRATION")
        print("-" * 30)
        
        for offset in [0, 100, 200]:
            enroll_data = self.fetch_data('enrollment', state='Maharashtra',
                                        limit=50, offset=offset)
            if enroll_data:
                print(f"   ✅ Offset {offset}: Data fetched")
            else:
                print(f"   ❌ Offset {offset}: Failed")
        
        # 4. District-level data
        print("\n4. DISTRICT-LEVEL DATA FETCHING")
        print("-" * 30)
        
        district_tests = [
            ('Tamil Nadu', 'Madurai'),
            ('Maharashtra', 'Mumbai'),
            ('Karnataka', 'Bengaluru Urban')
        ]
        
        for state, district in district_tests:
            district_data = self.fetch_data('demographic', state=state, 
                                          district=district, limit=10)
            if district_data:
                print(f"   ✅ {state} - {district}: Success")
            else:
                print(f"   ❌ {state} - {district}: Failed")
    
    def save_sample_data(self):
        """
        Save sample API responses for analysis
        """
        print("\n💾 SAVING SAMPLE DATA")
        print("-" * 30)
        
        # Create samples directory
        os.makedirs('api_samples', exist_ok=True)
        
        sample_states = ['Tamil Nadu', 'Maharashtra', 'Karnataka']
        
        for state in sample_states:
            # Fetch and save demographic data
            demo_data = self.fetch_data('demographic', state=state, limit=100)
            if demo_data:
                with open(f'api_samples/{state.replace(" ", "_")}_demographic.json', 'w') as f:
                    json.dump(demo_data, f, indent=2)
                print(f"   ✅ Saved demographic data for {state}")
            
            # Fetch and save biometric data
            bio_data = self.fetch_data('biometric', state=state, limit=100)
            if bio_data:
                with open(f'api_samples/{state.replace(" ", "_")}_biometric.json', 'w') as f:
                    json.dump(bio_data, f, indent=2)
                print(f"   ✅ Saved biometric data for {state}")
        
        print("   📁 Sample data saved in 'api_samples' directory")

def main():
    """
    Main function to demonstrate API integration
    """
    print("Note: This script demonstrates API integration patterns.")
    print("You'll need to replace the base_url with the actual API endpoint.")
    print("The current implementation uses placeholder URLs.\n")
    
    # Initialize API client
    client = AadhaarAPIClient()
    
    # Run demonstration
    client.demonstrate_api_usage()
    
    # Save sample data (if API is available)
    # client.save_sample_data()
    
    print("\n" + "="*50)
    print("API Integration demonstration completed!")
    print("\nTo use with real API:")
    print("1. Replace base_url in AadhaarAPIClient.__init__()")
    print("2. Verify API endpoints in api.md")
    print("3. Test with small data samples first")
    print("4. Implement proper error handling and rate limiting")

if __name__ == "__main__":
    main()