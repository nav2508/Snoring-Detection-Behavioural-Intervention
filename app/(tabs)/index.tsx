import { Link } from 'expo-router';
import { StyleSheet, View, Text, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

export default function HomeScreen() {
  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Ionicons name="bed" size={64} color="#007AFF" />
        <Text style={styles.title}>Snoring Analyzer</Text>
        <Text style={styles.subtitle}>
          Analyze your sleep patterns and detect snoring using AI
        </Text>
      </View>

      <View style={styles.features}>
        <View style={styles.featureItem}>
          <Ionicons name="cloud-upload" size={32} color="#007AFF" />
          <Text style={styles.featureTitle}>Upload MP3</Text>
          <Text style={styles.featureDescription}>
            Upload your recorded sleep audio in MP3 format
          </Text>
        </View>

        <View style={styles.featureItem}>
          <Ionicons name="analytics" size={32} color="#007AFF" />
          <Text style={styles.featureTitle}>AI Analysis</Text>
          <Text style={styles.featureDescription}>
            Advanced machine learning detects snoring patterns
          </Text>
        </View>

        <View style={styles.featureItem}>
          <Ionicons name="stats-chart" size={32} color="#007AFF" />
          <Text style={styles.featureTitle}>Detailed Report</Text>
          <Text style={styles.featureDescription}>
            Get comprehensive analysis with timestamps and statistics
          </Text>
        </View>
      </View>

      <Link href="/(tabs)/explore" asChild>
        <TouchableOpacity style={styles.ctaButton}>
          <Ionicons name="mic" size={20} color="white" />
          <Text style={styles.ctaButtonText}>Start Analysis</Text>
        </TouchableOpacity>
      </Link>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  header: {
    alignItems: 'center',
    paddingVertical: 40,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    marginTop: 20,
    color: '#333',
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginTop: 10,
    lineHeight: 22,
  },
  features: {
    marginVertical: 30,
  },
  featureItem: {
    backgroundColor: 'white',
    padding: 20,
    borderRadius: 12,
    marginBottom: 15,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  featureTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginTop: 10,
    color: '#333',
  },
  featureDescription: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginTop: 5,
    lineHeight: 20,
  },
  ctaButton: {
    backgroundColor: '#007AFF',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 10,
    marginTop: 20,
  },
  ctaButtonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
});