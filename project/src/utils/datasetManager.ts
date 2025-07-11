// Dataset Management Utilities for SignSpeak
// This file handles dataset operations and validation

export interface DatasetInfo {
  signLanguage: 'ASL' | 'ISL' | 'BSL';
  signName: string;
  imageCount: number;
  imagePaths: string[];
}

export interface DatasetStats {
  totalSigns: number;
  totalImages: number;
  signLanguages: {
    ASL: { signs: number; images: number };
    ISL: { signs: number; images: number };
    BSL: { signs: number; images: number };
  };
  signDetails: DatasetInfo[];
}

// Supported image formats
export const SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.webp'];

// Minimum images required per sign for training
export const MIN_IMAGES_PER_SIGN = 10;
export const RECOMMENDED_IMAGES_PER_SIGN = 50;

// Dataset folder structure
export const DATASET_STRUCTURE = {
  ASL: [
    'hello', 'goodbye', 'thank_you', 'please', 'sorry', 
    'yes', 'no', 'water', 'help', 'love'
  ],
  ISL: [
    'namaste', 'dhanyawad', 'paani', 'madad', 'ghar',
    'mata', 'pita', 'khana', 'samjha', 'aadab'
  ],
  BSL: [
    'hello', 'goodbye', 'please', 'thank_you', 'sorry',
    'water', 'help', 'mother', 'father', 'yes'
  ]
};

// Validate dataset structure and return statistics
export const validateDataset = async (): Promise<DatasetStats> => {
  const stats: DatasetStats = {
    totalSigns: 0,
    totalImages: 0,
    signLanguages: {
      ASL: { signs: 0, images: 0 },
      ISL: { signs: 0, images: 0 },
      BSL: { signs: 0, images: 0 }
    },
    signDetails: []
  };

  // In a real implementation, this would scan the actual filesystem
  // For now, we'll return mock data structure
  
  Object.entries(DATASET_STRUCTURE).forEach(([lang, signs]) => {
    const signLanguage = lang as 'ASL' | 'ISL' | 'BSL';
    
    signs.forEach(signName => {
      const mockImageCount = 0; // Will be replaced with actual file count
      const mockImagePaths: string[] = []; // Will be replaced with actual file paths
      
      const signInfo: DatasetInfo = {
        signLanguage,
        signName,
        imageCount: mockImageCount,
        imagePaths: mockImagePaths
      };
      
      stats.signDetails.push(signInfo);
      stats.signLanguages[signLanguage].signs++;
      stats.signLanguages[signLanguage].images += mockImageCount;
      stats.totalImages += mockImageCount;
    });
    
    stats.totalSigns += signs.length;
  });

  return stats;
};

// Check if dataset is ready for training
export const isDatasetReady = (stats: DatasetStats): boolean => {
  const hasMinimumImages = stats.signDetails.every(
    sign => sign.imageCount >= MIN_IMAGES_PER_SIGN
  );
  
  const hasAllSigns = stats.totalSigns > 0;
  
  return hasMinimumImages && hasAllSigns;
};

// Get dataset readiness report
export const getDatasetReport = (stats: DatasetStats): string => {
  const readySigns = stats.signDetails.filter(
    sign => sign.imageCount >= MIN_IMAGES_PER_SIGN
  ).length;
  
  const totalSigns = stats.signDetails.length;
  const readyPercentage = Math.round((readySigns / totalSigns) * 100);
  
  return `Dataset Status: ${readySigns}/${totalSigns} signs ready (${readyPercentage}%)`;
};

// Generate training recommendations
export const getTrainingRecommendations = (stats: DatasetStats): string[] => {
  const recommendations: string[] = [];
  
  stats.signDetails.forEach(sign => {
    if (sign.imageCount === 0) {
      recommendations.push(`❌ ${sign.signLanguage}/${sign.signName}: No images found`);
    } else if (sign.imageCount < MIN_IMAGES_PER_SIGN) {
      recommendations.push(
        `⚠️ ${sign.signLanguage}/${sign.signName}: Only ${sign.imageCount} images (need ${MIN_IMAGES_PER_SIGN} minimum)`
      );
    } else if (sign.imageCount < RECOMMENDED_IMAGES_PER_SIGN) {
      recommendations.push(
        `📈 ${sign.signLanguage}/${sign.signName}: ${sign.imageCount} images (${RECOMMENDED_IMAGES_PER_SIGN} recommended for better accuracy)`
      );
    } else {
      recommendations.push(
        `✅ ${sign.signLanguage}/${sign.signName}: ${sign.imageCount} images (ready for training)`
      );
    }
  });
  
  return recommendations;
};