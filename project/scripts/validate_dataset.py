#!/usr/bin/env python3
"""
Dataset Validation Script for SignSpeak
Validates the ASL dataset structure and provides statistics
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import argparse

def validate_dataset(dataset_path="./dataset"):
    """Validate the dataset structure and return statistics"""
    
    print("🔍 Validating SignSpeak Dataset...")
    print("=" * 50)
    
    dataset_dir = Path(dataset_path)
    asl_dir = dataset_dir / "ASL"
    
    if not asl_dir.exists():
        print("❌ ASL directory not found!")
        return False
    
    # Expected classes for ASL dataset
    expected_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    expected_numbers = [str(i) for i in range(10)]
    expected_classes = expected_letters + expected_numbers
    
    # Scan actual classes
    actual_classes = []
    class_stats = {}
    total_images = 0
    
    for class_dir in asl_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            actual_classes.append(class_name)
            
            # Count images in this class
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
            images = [f for f in class_dir.iterdir() 
                     if f.suffix.lower() in image_extensions]
            
            class_stats[class_name] = len(images)
            total_images += len(images)
    
    # Sort classes for better display
    actual_classes.sort()
    
    # Print statistics
    print(f"📊 Dataset Statistics:")
    print(f"   Total Classes: {len(actual_classes)}")
    print(f"   Total Images: {total_images}")
    print(f"   Average Images per Class: {total_images / len(actual_classes):.1f}")
    print()
    
    # Class distribution
    print("📋 Class Distribution:")
    print("-" * 30)
    
    letters_found = []
    numbers_found = []
    other_classes = []
    
    for class_name in actual_classes:
        count = class_stats[class_name]
        status = "✅" if count >= 50 else "⚠️" if count >= 10 else "❌"
        
        print(f"   {status} {class_name}: {count} images")
        
        if class_name in expected_letters:
            letters_found.append(class_name)
        elif class_name in expected_numbers:
            numbers_found.append(class_name)
        else:
            other_classes.append(class_name)
    
    print()
    
    # Coverage analysis
    print("📈 Coverage Analysis:")
    print("-" * 30)
    print(f"   Letters found: {len(letters_found)}/26")
    print(f"   Numbers found: {len(numbers_found)}/10")
    
    missing_letters = set(expected_letters) - set(letters_found)
    missing_numbers = set(expected_numbers) - set(numbers_found)
    
    if missing_letters:
        print(f"   Missing letters: {', '.join(sorted(missing_letters))}")
    
    if missing_numbers:
        print(f"   Missing numbers: {', '.join(sorted(missing_numbers))}")
    
    if other_classes:
        print(f"   Unexpected classes: {', '.join(other_classes)}")
    
    print()
    
    # Quality assessment
    print("🎯 Quality Assessment:")
    print("-" * 30)
    
    excellent_classes = [c for c, count in class_stats.items() if count >= 100]
    good_classes = [c for c, count in class_stats.items() if 50 <= count < 100]
    fair_classes = [c for c, count in class_stats.items() if 10 <= count < 50]
    poor_classes = [c for c, count in class_stats.items() if count < 10]
    
    print(f"   Excellent (≥100 images): {len(excellent_classes)} classes")
    print(f"   Good (50-99 images): {len(good_classes)} classes")
    print(f"   Fair (10-49 images): {len(fair_classes)} classes")
    print(f"   Poor (<10 images): {len(poor_classes)} classes")
    
    if poor_classes:
        print(f"   ⚠️ Classes needing more data: {', '.join(poor_classes)}")
    
    # Training readiness
    ready_for_training = len(poor_classes) == 0
    print()
    print("🚀 Training Readiness:")
    print("-" * 30)
    
    if ready_for_training:
        print("   ✅ Dataset is ready for training!")
        print("   All classes have sufficient images (≥10)")
    else:
        print("   ⚠️ Dataset needs more images for some classes")
        print(f"   {len(poor_classes)} classes have <10 images")
    
    # Save validation report
    report = {
        "validation_date": str(Path().resolve()),
        "total_classes": len(actual_classes),
        "total_images": total_images,
        "average_images_per_class": total_images / len(actual_classes),
        "class_stats": class_stats,
        "coverage": {
            "letters_found": len(letters_found),
            "numbers_found": len(numbers_found),
            "missing_letters": list(missing_letters),
            "missing_numbers": list(missing_numbers)
        },
        "quality": {
            "excellent": len(excellent_classes),
            "good": len(good_classes),
            "fair": len(fair_classes),
            "poor": len(poor_classes)
        },
        "ready_for_training": ready_for_training
    }
    
    report_path = dataset_dir / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Validation report saved to: {report_path}")
    
    return ready_for_training

def main():
    parser = argparse.ArgumentParser(description='Validate SignSpeak dataset')
    parser.add_argument('--dataset-path', default='./dataset', 
                       help='Path to dataset directory')
    
    args = parser.parse_args()
    
    success = validate_dataset(args.dataset_path)
    
    if success:
        print("\n🎉 Validation completed successfully!")
    else:
        print("\n❌ Validation found issues that need attention")
        exit(1)

if __name__ == "__main__":
    main()