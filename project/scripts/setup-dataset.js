#!/usr/bin/env node

/**
 * Dataset Setup Script for SignSpeak
 * Downloads and organizes the Kaggle ASL dataset
 */

import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

const DATASET_URL = 'ayuraj/asl-dataset';
const DATASET_DIR = './dataset';
const ASL_DIR = './dataset/ASL';

console.log('🚀 Setting up SignSpeak ASL Dataset...\n');

// Check if Kaggle CLI is installed
function checkKaggleCLI() {
  try {
    execSync('kaggle --version', { stdio: 'pipe' });
    console.log('✅ Kaggle CLI found');
    return true;
  } catch (error) {
    console.log('❌ Kaggle CLI not found');
    console.log('\n📋 To install Kaggle CLI:');
    console.log('1. pip install kaggle');
    console.log('2. Create ~/.kaggle/kaggle.json with your API credentials');
    console.log('3. chmod 600 ~/.kaggle/kaggle.json');
    console.log('\n🔗 Get API credentials from: https://www.kaggle.com/settings/account');
    return false;
  }
}

// Create directory structure
function createDirectories() {
  console.log('📁 Creating directory structure...');
  
  if (!fs.existsSync(DATASET_DIR)) {
    fs.mkdirSync(DATASET_DIR, { recursive: true });
  }
  
  if (!fs.existsSync(ASL_DIR)) {
    fs.mkdirSync(ASL_DIR, { recursive: true });
  }
  
  console.log('✅ Directories created');
}

// Download dataset from Kaggle
function downloadDataset() {
  console.log('⬇️ Downloading ASL dataset from Kaggle...');
  console.log('This may take a few minutes...\n');
  
  try {
    // Download the dataset
    execSync(`kaggle datasets download -d ${DATASET_URL} -p ${DATASET_DIR}`, { 
      stdio: 'inherit' 
    });
    
    console.log('✅ Dataset downloaded successfully');
    return true;
  } catch (error) {
    console.error('❌ Failed to download dataset:', error.message);
    return false;
  }
}

// Extract and organize dataset
function extractDataset() {
  console.log('📦 Extracting dataset...');
  
  const zipFile = path.join(DATASET_DIR, 'asl-dataset.zip');
  
  if (!fs.existsSync(zipFile)) {
    console.error('❌ Dataset zip file not found');
    return false;
  }
  
  try {
    // Extract the zip file
    execSync(`cd ${DATASET_DIR} && unzip -q asl-dataset.zip`, { stdio: 'inherit' });
    
    // The dataset typically extracts to a folder like 'asl_dataset' or similar
    // We need to find the extracted folder and move its contents to ASL/
    const extractedDirs = fs.readdirSync(DATASET_DIR).filter(item => {
      const itemPath = path.join(DATASET_DIR, item);
      return fs.statSync(itemPath).isDirectory() && item !== 'ASL';
    });
    
    if (extractedDirs.length > 0) {
      const sourceDir = path.join(DATASET_DIR, extractedDirs[0]);
      console.log(`📂 Found extracted directory: ${extractedDirs[0]}`);
      
      // Move contents to ASL directory
      const items = fs.readdirSync(sourceDir);
      items.forEach(item => {
        const sourcePath = path.join(sourceDir, item);
        const destPath = path.join(ASL_DIR, item);
        
        if (fs.statSync(sourcePath).isDirectory()) {
          execSync(`cp -r "${sourcePath}" "${destPath}"`, { stdio: 'inherit' });
        }
      });
      
      // Clean up
      execSync(`rm -rf "${sourceDir}"`, { stdio: 'inherit' });
    }
    
    // Remove zip file
    fs.unlinkSync(zipFile);
    
    console.log('✅ Dataset extracted and organized');
    return true;
  } catch (error) {
    console.error('❌ Failed to extract dataset:', error.message);
    return false;
  }
}

// Validate dataset structure
function validateDataset() {
  console.log('🔍 Validating dataset structure...');
  
  if (!fs.existsSync(ASL_DIR)) {
    console.error('❌ ASL directory not found');
    return false;
  }
  
  const expectedClasses = [
    // Letters A-Z
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    // Numbers 0-9
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
  ];
  
  const actualClasses = fs.readdirSync(ASL_DIR).filter(item => {
    const itemPath = path.join(ASL_DIR, item);
    return fs.statSync(itemPath).isDirectory();
  });
  
  console.log(`📊 Found ${actualClasses.length} classes`);
  
  let totalImages = 0;
  const classStats = [];
  
  actualClasses.forEach(className => {
    const classPath = path.join(ASL_DIR, className);
    const images = fs.readdirSync(classPath).filter(file => 
      /\.(jpg|jpeg|png|webp)$/i.test(file)
    );
    
    totalImages += images.length;
    classStats.push({ class: className, images: images.length });
  });
  
  console.log(`📈 Total images: ${totalImages}`);
  console.log('📋 Class distribution:');
  
  classStats.sort((a, b) => a.class.localeCompare(b.class));
  classStats.forEach(stat => {
    console.log(`   ${stat.class}: ${stat.images} images`);
  });
  
  // Check for missing expected classes
  const missingClasses = expectedClasses.filter(cls => !actualClasses.includes(cls));
  if (missingClasses.length > 0) {
    console.log(`⚠️ Missing classes: ${missingClasses.join(', ')}`);
  }
  
  console.log('✅ Dataset validation complete');
  return true;
}

// Create dataset info file
function createDatasetInfo() {
  console.log('📝 Creating dataset info file...');
  
  const info = {
    name: 'ASL Dataset',
    source: 'https://www.kaggle.com/datasets/ayuraj/asl-dataset',
    description: 'American Sign Language letters (A-Z) and numbers (0-9)',
    classes: 36,
    totalImages: 0,
    lastUpdated: new Date().toISOString(),
    structure: {
      'ASL/': 'American Sign Language images organized by class'
    }
  };
  
  // Count actual images
  if (fs.existsSync(ASL_DIR)) {
    const classes = fs.readdirSync(ASL_DIR).filter(item => {
      const itemPath = path.join(ASL_DIR, item);
      return fs.statSync(itemPath).isDirectory();
    });
    
    classes.forEach(className => {
      const classPath = path.join(ASL_DIR, className);
      const images = fs.readdirSync(classPath).filter(file => 
        /\.(jpg|jpeg|png|webp)$/i.test(file)
      );
      info.totalImages += images.length;
    });
  }
  
  fs.writeFileSync(
    path.join(DATASET_DIR, 'dataset-info.json'), 
    JSON.stringify(info, null, 2)
  );
  
  console.log('✅ Dataset info file created');
}

// Main execution
async function main() {
  try {
    // Step 1: Check prerequisites
    if (!checkKaggleCLI()) {
      process.exit(1);
    }
    
    // Step 2: Create directories
    createDirectories();
    
    // Step 3: Download dataset
    if (!downloadDataset()) {
      process.exit(1);
    }
    
    // Step 4: Extract and organize
    if (!extractDataset()) {
      process.exit(1);
    }
    
    // Step 5: Validate
    if (!validateDataset()) {
      process.exit(1);
    }
    
    // Step 6: Create info file
    createDatasetInfo();
    
    console.log('\n🎉 Dataset setup complete!');
    console.log('\n📋 Next steps:');
    console.log('1. Run: npm run validate-dataset');
    console.log('2. Run: npm run train-model');
    console.log('3. Test the model in the app');
    
  } catch (error) {
    console.error('❌ Setup failed:', error.message);
    process.exit(1);
  }
}

main();