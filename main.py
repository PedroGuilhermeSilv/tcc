from converter.services.export_img import VideoFrameExtractor
from object_values.type_videos import TypeVideos


def main():
    path_video = "src/tmp/videos"
    name_video = "IMG_1115"
    format = TypeVideos.MOV
    path_image_metadata = "src/tmp/base.JPEG"
    export_img = VideoFrameExtractor(
        path_video, path_image_metadata, name_video, format
    )
    export_img.execute()


if __name__ == "__main__":
    main()
