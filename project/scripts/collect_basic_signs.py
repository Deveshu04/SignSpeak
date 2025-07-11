#!/usr/bin/env python3
"""
Basic Sign Data Collection Script for SignSpeak
Collects webcam data for basic ASL signs to create a training dataset
"""

import cv2
import os
import numpy as np
from pathlib import Path
import time
import json

class BasicSignCollector:
    def __init__(self, data_dir="Data"):
        self.data_dir = Path(data_dir)
        self.asl_dir = self.data_dir / "asl_dataset"
        self.asl_dir.mkdir(parents=True, exist_ok=True)

        # Start with basic signs - common letters and numbers
        self.basic_signs = ['a', 'b', 'c', 'd', 'e', 'hello', 'thank_you', 'please', '1', '2', '3', '4', '5']

        # Create directories for each sign
        for sign in self.basic_signs:
            sign_dir = self.asl_dir / sign
            sign_dir.mkdir(exist_ok=True)

        self.current_sign = None
        self.images_per_sign = 100
        self.countdown_duration = 3

    def collect_sign_data(self, sign_name):
        """Collect images for a specific sign"""
        if sign_name not in self.basic_signs:
            print(f"❌ Sign '{sign_name}' not in basic signs list")
            return

        sign_dir = self.asl_dir / sign_name
        existing_images = len(list(sign_dir.glob("*.jpg")))

        print(f"\n📸 Collecting data for sign: '{sign_name.upper()}'")
        print(f"   Target: {self.images_per_sign} images")
        print(f"   Existing: {existing_images} images")
        print(f"   To collect: {max(0, self.images_per_sign - existing_images)} images")

        if existing_images >= self.images_per_sign:
            print(f"✅ Sufficient data already collected for '{sign_name}'")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        collected = existing_images
        collecting = False

        print(f"\n🎥 Camera ready! Instructions:")
        print(f"   - Position your hand to show the '{sign_name.upper()}' sign")
        print(f"   - Press SPACE to start collecting")
        print(f"   - Press 'q' to quit")
        print(f"   - Keep your hand steady while collecting")

        while collected < self.images_per_sign:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Draw UI elements
            cv2.putText(frame, f"Sign: {sign_name.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Collected: {collected}/{self.images_per_sign}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if collecting:
                cv2.putText(frame, "COLLECTING...", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Save the frame
                filename = sign_dir / f"{sign_name}_{collected:04d}.jpg"
                cv2.imwrite(str(filename), frame)
                collected += 1

                # Small delay between captures
                time.sleep(0.1)

                if collected >= self.images_per_sign:
                    print(f"✅ Completed collecting {self.images_per_sign} images for '{sign_name}'")
                    break
            else:
                cv2.putText(frame, "Press SPACE to start", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Draw a rectangle for hand positioning
            cv2.rectangle(frame, (200, 100), (440, 340), (255, 255, 255), 2)
            cv2.putText(frame, "Position hand here", (210, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Sign Collection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if not collecting:
                    collecting = True
                    print(f"🚀 Started collecting images for '{sign_name}'...")
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        print(f"✅ Finished collecting for '{sign_name}': {collected} images total")

    def collect_all_basic_signs(self):
        """Collect data for all basic signs"""
        print("🎯 Starting basic sign data collection")
        print(f"   Signs to collect: {', '.join(self.basic_signs)}")

        for i, sign in enumerate(self.basic_signs):
            print(f"\n{'='*50}")
            print(f"Progress: {i+1}/{len(self.basic_signs)}")

            try:
                self.collect_sign_data(sign)

                if i < len(self.basic_signs) - 1:
                    print(f"\n⏳ Get ready for next sign in 3 seconds...")
                    time.sleep(3)

            except KeyboardInterrupt:
                print(f"\n🛑 Collection interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error collecting '{sign}': {e}")
                continue

        print(f"\n🎉 Basic sign collection completed!")
        self.show_collection_summary()

    def show_collection_summary(self):
        """Show summary of collected data"""
        print(f"\n📊 Collection Summary:")
        print(f"{'Sign':<15} {'Images':<8} {'Status'}")
        print("-" * 35)

        total_images = 0
        for sign in self.basic_signs:
            sign_dir = self.asl_dir / sign
            count = len(list(sign_dir.glob("*.jpg")))
            total_images += count
            status = "✅ Complete" if count >= self.images_per_sign else f"⚠️ Need {self.images_per_sign - count} more"
            print(f"{sign:<15} {count:<8} {status}")

        print("-" * 35)
        print(f"Total images: {total_images}")

        # Save collection info
        info = {
            "signs": self.basic_signs,
            "total_images": total_images,
            "images_per_sign_target": self.images_per_sign,
            "collection_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(self.data_dir / "collection_info.json", 'w') as f:
            json.dump(info, f, indent=2)

if __name__ == "__main__":
    collector = BasicSignCollector()

    print("🚀 SignSpeak Basic Sign Data Collector")
    print("=" * 50)

    while True:
        print("\nOptions:")
        print("1. Collect all basic signs")
        print("2. Collect specific sign")
        print("3. Show collection summary")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == '1':
            collector.collect_all_basic_signs()
        elif choice == '2':
            print(f"Available signs: {', '.join(collector.basic_signs)}")
            sign = input("Enter sign name: ").strip().lower()
            collector.collect_sign_data(sign)
        elif choice == '3':
            collector.show_collection_summary()
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")
