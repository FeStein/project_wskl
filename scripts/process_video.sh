filename="video_1"

ffprobe -v quiet -print_format json -show_format -show_streams video_1.mov > "${filename}_raw.dat"

#change filetype .mov to .mp4
ffmpeg -i "${filename}.mov" -vcodec h264 -acodec mp2 -filter:v fps=fps=10 "tmp.mp4"

#scale resolution to 720p
ffmpeg -i "tmp.mp4" -vf scale=1280:720 "${filename}_p.mp4"

#remove temporary files
rm "tmp.mp4"

