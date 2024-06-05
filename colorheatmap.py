import numpy as np
import cv2
import copy
import os
from make_video import make_video
from progress.bar import Bar

def main():
    # Ensure the frames directory exists
    if not os.path.exists('./frames'):
        os.makedirs('./frames')

    capture = cv2.VideoCapture('input.mp4')
    if not capture.isOpened():
        print("Error: Could not open video.")
        return

    background_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of frames: {length}")

    bar = Bar('Processing Frames', max=length)

    first_iteration_indicator = True
    accum_image = None
    first_frame = None

    for i in range(0, length):
        ret, frame = capture.read()
        if not ret:
            print(f"Error: Frame {i} could not be read.")
            break

        if first_iteration_indicator:
            first_frame = copy.deepcopy(frame)
            height, width = frame.shape[:2]
            accum_image = np.zeros((height, width), np.uint8)
            first_iteration_indicator = False
            print(f"Initialized accum_image with shape: {accum_image.shape}")
        else:
            filter = background_subtractor.apply(frame)  # remove the background
            cv2.imwrite('./frame.jpg', frame)
            cv2.imwrite('./diff-bkgnd-frame.jpg', filter)

            threshold = 2
            maxValue = 2
            _, th1 = cv2.threshold(filter, threshold, maxValue, cv2.THRESH_BINARY)

            # add to the accumulated image
            accum_image = cv2.add(accum_image, th1)
            cv2.imwrite('./mask.jpg', accum_image)

            # Apply colormap for different zones
            color_image_video = np.zeros_like(accum_image, dtype=np.uint8)
            color_image_video[accum_image < 50] = 0  # Blue zone
            color_image_video[(accum_image >= 50) & (accum_image < 100)] = 85  # Green zone
            color_image_video[(accum_image >= 100) & (accum_image < 150)] = 170  # Yellow zone
            color_image_video[accum_image >= 150] = 255  # Red zone

            video_frame = cv2.applyColorMap(color_image_video, cv2.COLORMAP_HOT)  # You can change COLORMAP_HOT to other colormaps

            video_frame = cv2.addWeighted(frame, 0.7, video_frame, 0.7, 0)

            name = "./frames/frame%d.jpg" % i
            cv2.imwrite(name, video_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        bar.next()

    bar.finish()

    if first_frame is not None and accum_image is not None:
        print("Creating video...")
        make_video('./frames/', './output.avi')

        # Overlay different zones on the first frame
        color_image = np.zeros_like(accum_image, dtype=np.uint8)
        color_image[accum_image < 50] = 0  # Blue zone
        color_image[(accum_image >= 50) & (accum_image < 100)] = 85  # Green zone
        color_image[(accum_image >= 100) & (accum_image < 150)] = 170  # Yellow zone
        color_image[accum_image >= 150] = 255  # Red zone

        color_image = cv2.applyColorMap(color_image, cv2.COLORMAP_HOT)  # You can change COLORMAP_HOT to other colormaps
        result_overlay = cv2.addWeighted(first_frame, 0.7, color_image, 0.7, 0)

        # Save the final heatmap
        cv2.imwrite('diff-overlay.jpg', result_overlay)
        print("Saved final heatmap as 'diff-overlay.jpg'")

    # Cleanup
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
