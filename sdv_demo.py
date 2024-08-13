from sdv.datasets.demo import download_demo
from sdv.single_table import GaussianCopulaSynthesizer

# Download demo data
real_data, metadata = download_demo('single_table', 'fake_hotel_guests')

# Initialize the synthesizer
synthesizer = GaussianCopulaSynthesizer(metadata)

# Fit the synthesizer to the real data
synthesizer.fit(real_data)

# Sample synthetic data
synthetic_data = synthesizer.sample(num_rows=10)

print(synthetic_data)
