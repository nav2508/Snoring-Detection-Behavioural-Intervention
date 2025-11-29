import { Ionicons } from '@expo/vector-icons';
import axios from 'axios';
import * as DocumentPicker from 'expo-document-picker';
import { useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View
} from 'react-native';

// ✅ Use your computer's IP address
const API_BASE_URL = 'http://192.168.1.83:8000';


// Function to convert base64 to image source
const base64ToImageSource = (base64String: string) => {
  return { uri: `data:image/png;base64,${base64String}` };
};

export default function ExploreScreen() {
  const [selectedFile, setSelectedFile] = useState<DocumentPicker.DocumentPickerResult | null>(null);
  const [results, setResults] = useState<any>(null);
  const [visualizations, setVisualizations] = useState<{[key: string]: string} | null>(null);
  const [loading, setLoading] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'unknown' | 'connected' | 'failed'>('unknown');

  const pickAudioFile = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: 'audio/mpeg',
        copyToCacheDirectory: true,
      });

      if (result.assets && result.assets.length > 0) {
        setSelectedFile(result);
        setResults(null);
        setVisualizations(null);
        console.log('Selected file:', result.assets[0].name);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to pick file');
      console.error('File pick error:', error);
    }
  };

  const testConnection = async () => {
    try {
      console.log('Testing connection to:', `${API_BASE_URL}/health`);
      setConnectionStatus('unknown');
      
      const response = await axios.get(`${API_BASE_URL}/health`, {
        timeout: 5000,
      });
      
      console.log('✅ Connection test successful:', response.data);
      setConnectionStatus('connected');
      Alert.alert('Success', 'Connected to server successfully!');
      return true;
    } catch (error: any) {
      console.error('❌ Connection test failed:', error);
      setConnectionStatus('failed');
      
      let errorMessage = 'Cannot connect to server. Please check:\n\n';
      
      if (error.code === 'ECONNREFUSED') {
        errorMessage += '• Backend is not running\n• Port 8000 is blocked\n• Wrong IP address';
      } else if (error.message?.includes('Network Error')) {
        errorMessage += '• Devices not on same WiFi\n• Firewall blocking connection\n• Wrong IP address';
      } else {
        errorMessage += error.message || 'Unknown error occurred';
      }
      
      Alert.alert('Connection Failed', errorMessage);
      return false;
    }
  };

  const uploadAudio = async () => {
    if (!selectedFile?.assets?.[0]) {
      Alert.alert('Error', 'Please select an MP3 file first');
      return;
    }

    setLoading(true);
    setResults(null);
    setVisualizations(null);

    try {
      // Test connection first
      console.log('Testing connection before upload...');
      const isConnected = await testConnection();
      if (!isConnected) {
        throw new Error('Cannot connect to server. Please check connection and try again.');
      }

      const formData = new FormData();
      const file = selectedFile.assets[0];
      
      // @ts-ignore - Expo types issue with FormData
      formData.append('file', {
        uri: file.uri,
        type: 'audio/mpeg',
        name: file.name,
      });

      console.log('Starting upload to:', `${API_BASE_URL}/analyze-snoring`);
      console.log('File:', file.name, 'Size:', file.size);

      const response = await axios.post(`${API_BASE_URL}/analyze-snoring`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000, // 2 minutes timeout
      });

      console.log('✅ Upload successful, response received');
      setResults(response.data);
      
      // Set visualizations if available
      if (response.data.visualizations) {
        setVisualizations(response.data.visualizations);
      }

    } catch (error: any) {
      console.error('❌ Upload failed:', error);
      
      let errorMessage = 'Upload failed. ';
      
      if (error.code === 'ECONNREFUSED') {
        errorMessage = `Cannot connect to server at ${API_BASE_URL}. Please check:\n\n1. Backend is running\n2. Both devices on same WiFi\n3. Firewall allows Python\n4. Correct IP address: 192.168.0.101`;
      } else if (error.response) {
        errorMessage = `Server error: ${error.response.data?.detail || 'Unknown server error'}`;
      } else if (error.request) {
        errorMessage = 'No response from server. Check your connection.';
      } else if (error.message) {
        errorMessage = error.message;
      }

      Alert.alert('Upload Failed', errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return '#4CAF50';
      case 'failed': return '#F44336';
      default: return '#FF9800';
    }
  };

  const getConnectionStatusText = () => {
    switch (connectionStatus) {
      case 'connected': return 'Connected to Server';
      case 'failed': return 'Connection Failed';
      default: return 'Not Tested';
    }
  };

  const getSnoringLevel = (ratio: number) => {
    if (ratio > 0.3) return { text: 'HIGH', color: '#ff4444', icon: 'warning' };
    if (ratio > 0.1) return { text: 'MODERATE', color: '#ffaa00', icon: 'alert' };
    return { text: 'LOW', color: '#00aa00', icon: 'checkmark' };
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Ionicons name="mic" size={32} color="#007AFF" />
        <Text style={styles.title}>Snoring Detection</Text>
        <Text style={styles.subtitle}>Upload an MP3 file to analyze snoring patterns</Text>
        
        {/* Connection Info */}
        <View style={styles.connectionSection}>
          <View style={styles.connectionRow}>
            <Ionicons name="server" size={16} color="#666" />
            <Text style={styles.connectionText}>Server: {API_BASE_URL}</Text>
          </View>
          
          <View style={styles.connectionRow}>
            <View 
              style={[
                styles.statusIndicator, 
                { backgroundColor: getConnectionStatusColor() }
              ]} 
            />
            <Text style={styles.statusText}>
              Status: {getConnectionStatusText()}
            </Text>
          </View>
        </View>

        {/* Test Connection Button */}
        <TouchableOpacity 
          style={styles.testButton}
          onPress={testConnection}
        >
          <Ionicons name="wifi" size={16} color="white" />
          <Text style={styles.testButtonText}>Test Connection</Text>
        </TouchableOpacity>
      </View>

      {/* File Selection */}
      <View style={styles.uploadSection}>
        <TouchableOpacity style={styles.uploadButton} onPress={pickAudioFile}>
          <Ionicons name="cloud-upload" size={24} color="white" />
          <Text style={styles.buttonText}>
            {selectedFile ? 'Change MP3 File' : 'Select MP3 File'}
          </Text>
        </TouchableOpacity>

        {selectedFile?.assets?.[0] && (
          <View style={styles.fileInfo}>
            <Ionicons name="musical-notes" size={20} color="#007AFF" />
            <Text style={styles.fileText} numberOfLines={1}>
              {selectedFile.assets[0].name}
            </Text>
          </View>
        )}
      </View>

      {/* Analyze Button */}
      <TouchableOpacity 
        style={[
          styles.analyzeButton, 
          (!selectedFile || loading) && styles.disabledButton
        ]} 
        onPress={uploadAudio}
        disabled={!selectedFile || loading}
      >
        {loading ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator color="#fff" />
            <Text style={styles.loadingText}>Analyzing...</Text>
          </View>
        ) : (
          <>
            <Ionicons name="analytics" size={20} color="white" />
            <Text style={styles.buttonText}>Analyze for Snoring</Text>
          </>
        )}
      </TouchableOpacity>

      {/* Results */}
      {results && (
        <View style={styles.resultsContainer}>
          <Text style={styles.resultsTitle}>Analysis Results</Text>
          
          {/* Overall Summary */}
          <View style={styles.summaryCard}>
            <View style={styles.summaryHeader}>
              <Ionicons 
                name={getSnoringLevel(results.analysis.snoring_ratio).icon as any} 
                size={24} 
                color={getSnoringLevel(results.analysis.snoring_ratio).color} 
              />
              <Text style={styles.summaryTitle}>Overall Summary</Text>
            </View>
            
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>Snoring Level:</Text>
              <Text style={[styles.summaryValue, { color: getSnoringLevel(results.analysis.snoring_ratio).color }]}>
                {getSnoringLevel(results.analysis.snoring_ratio).text}
              </Text>
            </View>
            
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>Snoring Ratio:</Text>
              <Text style={styles.summaryValue}>
                {(results.analysis.snoring_ratio * 100).toFixed(1)}%
              </Text>
            </View>

            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>Audio Duration:</Text>
              <Text style={styles.summaryValue}>
                {formatTime(results.audio_duration)}
              </Text>
            </View>
          </View>

          {/* Statistics */}
          <View style={styles.statsGrid}>
            <View style={styles.statCard}>
              <Text style={styles.statNumber}>{results.analysis.total_segments}</Text>
              <Text style={styles.statLabel}>Total Segments</Text>
            </View>
            <View style={styles.statCard}>
              <Text style={styles.statNumber}>{results.analysis.snoring_segments}</Text>
              <Text style={styles.statLabel}>Snoring Segments</Text>
            </View>
            <View style={styles.statCard}>
              <Text style={styles.statNumber}>{results.analysis.interval_count}</Text>
              <Text style={styles.statLabel}>Snoring Intervals</Text>
            </View>
          </View>

          {/* Snoring Intervals */}
          {results.analysis.snoring_intervals.length > 0 && (
            <View style={styles.intervalsContainer}>
              <Text style={styles.sectionTitle}>Snoring Intervals</Text>
              {results.analysis.snoring_intervals.map((interval: any, index: number) => (
                <View key={index} style={styles.intervalItem}>
                  <View style={styles.intervalHeader}>
                    <Ionicons name="time" size={16} color="#666" />
                    <Text style={styles.intervalTitle}>Interval {index + 1}</Text>
                  </View>
                  <Text style={styles.intervalText}>
                    {interval.start_time}s - {interval.end_time}s ({interval.duration}s)
                  </Text>
                </View>
              ))}
            </View>
          )}

          {/* Segment Predictions */}
          {results.segment_predictions && results.segment_predictions.length > 0 && (
            <View style={styles.segmentsContainer}>
              <Text style={styles.sectionTitle}>Segment Analysis</Text>
              {results.segment_predictions.slice(0, 8).map((pred: any, index: number) => (
                <View key={index} style={[
                  styles.segmentItem,
                  pred.class === 'Snoring' ? styles.snoringSegment : styles.normalSegment
                ]}>
                  <Text style={styles.segmentText}>
                    Segment {pred.segment}: {pred.class}
                  </Text>
                  <Text style={styles.confidenceText}>
                    {(pred.confidence * 100).toFixed(1)}% confidence
                  </Text>
                </View>
              ))}
            </View>
          )}

          {/* Conclusion Message */}
          <View style={styles.conclusionCard}>
            <Text style={styles.conclusionText}>{results.message}</Text>
          </View>
        </View>
      )}

      {/* Visualizations */}
      {visualizations && (
        <View style={styles.visualizationsContainer}>
          <Text style={styles.visualizationsTitle}>Analysis Visualizations</Text>
          
          {visualizations.analysis_plot && (
            <View style={styles.visualizationCard}>
              <Text style={styles.visualizationTitle}>Comprehensive Analysis</Text>
              <Image 
                source={base64ToImageSource(visualizations.analysis_plot)}
                style={styles.visualizationImage}
                resizeMode="contain"
              />
              <Text style={styles.visualizationDescription}>
                Shows audio waveform, snoring probability, detection timeline, and distribution
              </Text>
            </View>
          )}
          
          {visualizations.statistics_plot && (
            <View style={styles.visualizationCard}>
              <Text style={styles.visualizationTitle}>Statistics & Metrics</Text>
              <Image 
                source={base64ToImageSource(visualizations.statistics_plot)}
                style={styles.visualizationImage}
                resizeMode="contain"
              />
              <Text style={styles.visualizationDescription}>
                Detailed statistics, confidence distribution, and interval analysis
              </Text>
            </View>
          )}
          
          {visualizations.timeline_plot && (
            <View style={styles.visualizationCard}>
              <Text style={styles.visualizationTitle}>Snoring Timeline</Text>
              <Image 
                source={base64ToImageSource(visualizations.timeline_plot)}
                style={styles.visualizationImage}
                resizeMode="contain"
              />
              <Text style={styles.visualizationDescription}>
                Clear timeline showing snoring patterns throughout the recording
              </Text>
            </View>
          )}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    alignItems: 'center',
    padding: 20,
    backgroundColor: 'white',
    marginBottom: 10,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginTop: 10,
    color: '#333',
  },
  subtitle: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginTop: 5,
    marginBottom: 15,
  },
  connectionSection: {
    width: '100%',
    backgroundColor: '#f8f9fa',
    padding: 12,
    borderRadius: 8,
    marginBottom: 10,
  },
  connectionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 5,
  },
  connectionText: {
    fontSize: 12,
    color: '#666',
    marginLeft: 5,
  },
  statusIndicator: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },
  statusText: {
    fontSize: 12,
    color: '#666',
  },
  testButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#007AFF',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 6,
    gap: 6,
  },
  testButtonText: {
    color: 'white',
    fontSize: 14,
    fontWeight: 'bold',
  },
  uploadSection: {
    backgroundColor: 'white',
    padding: 20,
    marginHorizontal: 10,
    borderRadius: 12,
    marginBottom: 10,
  },
  uploadButton: {
    backgroundColor: '#007AFF',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 10,
  },
  analyzeButton: {
    backgroundColor: '#34C759',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    margin: 10,
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 10,
  },
  disabledButton: {
    backgroundColor: '#ccc',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  loadingText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  fileInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 10,
    padding: 10,
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    gap: 10,
  },
  fileText: {
    fontSize: 14,
    color: '#333',
    flex: 1,
  },
  resultsContainer: {
    padding: 10,
  },
  resultsTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 15,
    color: '#333',
  },
  summaryCard: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 12,
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  summaryHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
    gap: 10,
  },
  summaryTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 5,
  },
  summaryLabel: {
    fontSize: 16,
    color: '#666',
  },
  summaryValue: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10,
    gap: 10,
  },
  statCard: {
    flex: 1,
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  statNumber: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#007AFF',
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 5,
    textAlign: 'center',
  },
  intervalsContainer: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 12,
    marginBottom: 10,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
  },
  intervalItem: {
    padding: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#ff4444',
    marginBottom: 8,
    backgroundColor: '#fff5f5',
    borderRadius: 8,
  },
  intervalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 5,
    gap: 5,
  },
  intervalTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
  },
  intervalText: {
    fontSize: 12,
    color: '#666',
  },
  segmentsContainer: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 12,
    marginBottom: 10,
  },
  segmentItem: {
    padding: 10,
    borderRadius: 6,
    marginBottom: 5,
  },
  snoringSegment: {
    backgroundColor: '#fff5f5',
    borderLeftWidth: 4,
    borderLeftColor: '#ff4444',
  },
  normalSegment: {
    backgroundColor: '#f8f9fa',
    borderLeftWidth: 4,
    borderLeftColor: '#34C759',
  },
  segmentText: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#333',
  },
  confidenceText: {
    fontSize: 10,
    color: '#666',
    marginTop: 2,
  },
  conclusionCard: {
    backgroundColor: '#e3f2fd',
    padding: 15,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#2196f3',
  },
  conclusionText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1976d2',
    textAlign: 'center',
  },
  // Visualization Styles
  visualizationsContainer: {
    marginTop: 20,
    padding: 10,
  },
  visualizationsTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 15,
    color: '#333',
  },
  visualizationCard: {
    backgroundColor: 'white',
    padding: 15,
    borderRadius: 12,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  visualizationTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#333',
    textAlign: 'center',
  },
  visualizationImage: {
    width: '100%',
    height: 300,
    borderRadius: 8,
  },
  visualizationDescription: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
    marginTop: 8,
    fontStyle: 'italic',
  },
});