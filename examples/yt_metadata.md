### Download YouTube metadata & subtitles:
#### Usage

This is how you can download a list of youtube videos along with the associated youtube metadata using video2dataset. You must pass in a yt_metadata_args parameter which will be used to specify what information you want in your json metadata in the output dataset. Currently if you specify you want subtitles, all of the subtitles will get written to each samples metadata (subject to change soon).

```py
if __name__ == '__main__':

    yt_metadata_args = {
        'writesubtitles': True, # whether to write subtitles to a file
        'subtitleslangs': ['en'], # languages of subtitles (right now support only one language)
        'writeautomaticsub': True, # whether to write automatic subtitles
        'get_info': True # whether to save a video meta data into the output JSON file
    }

    video2dataset(
        url_list='input.parquet',
        input_format='parquet',
        output_format='files',
        output_folder='audio',
        yt_metadata_args=yt_metadata_args
    )
```

#### Output

For every sample the metadata will be present in the json file as such:

```json
{
    "url": "https://www.youtube.com/watch?v=q2ZOEFAaaI0",
    "key": "000000000",
    "status": "success",
    "error_message": null,
    "info": {
        "id": "q2ZOEFAaaI0",
        "title": "Q-learning with numpy and OpenAI Taxi-v2 \ud83d\ude95 (tutorial)",
        "thumbnail": "https://i.ytimg.com/vi_webp/q2ZOEFAaaI0/maxresdefault.webp",
        "description": "We'll train an Q-learning agent with Numpy that learns to play Taxi-v2. Where he must take a passenger at one location and drop him off at another as fast as possible. \ud83d\ude95\n\nThis video is part of the Deep Reinforcement Learning course with tensorflow \ud83d\udd79\ufe0f a free series of blog posts and videos \ud83c\udd95 about Deep Reinforcement Learning, where we'll learn the main algorithms, and how to implement them with Tensorflow : https://simoninithomas.github.io/Deep_reinforcement_learning_Course/\n\nIf you're new in Reinforcement Learning, please read first my article \"An introduction to Reinforcement Learning\": https://medium.freecodecamp.org/an-introduction-to-reinforcement-learning-4339519de419\n\nThe Q-learning article: https://medium.freecodecamp.org/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe\n\nThe Q-learning notebook: https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/Taxi-v2/Q%20Learning%20with%20OpenAI%20Taxi-v2%20video%20version.ipynb\n\nThis is my first video so if you have some feedbacks and advice please comment below.\n\nMoreover if you have some questions you can ask me in the comments.\n\nDon't forget to subscribe ! And to follow me on social media:\nTwitter: https://twitter.com/ThomasSimonini\nFacebook: https://www.facebook.com/thomas.simonini.3\n\nKeep learning, stay awesome!",
        "uploader": "Thomas Simonini",
        "uploader_id": "UC8XuSf1eD9AF8x8J19ha5og",
        "uploader_url": "http://www.youtube.com/channel/UC8XuSf1eD9AF8x8J19ha5og",
        "channel_id": "UC8XuSf1eD9AF8x8J19ha5og",
        "channel_url": "https://www.youtube.com/channel/UC8XuSf1eD9AF8x8J19ha5og",
        "duration": 776,
        "view_count": 44988,
        "average_rating": null,
        "age_limit": 0,
        "webpage_url": "https://www.youtube.com/watch?v=q2ZOEFAaaI0",
        "categories": [
            "Education"
        ],
        "tags": [
            "AI",
            "Q-learning",
            "tutorial",
            "numpy",
            "reinforcement-learning",
            "programming",
            "openai",
            "deep-learning",
            "Machine-learning"
        ],
        "playable_in_embed": true,
        "live_status": null,
        "release_timestamp": null,
        "comment_count": 126,
        "chapters": [
            ...
        ],
       ...
    }
}
```

Including the subtitles:
TODO(Move this into the previous block when we introduce clip=subtitle)

```json
"subtitles": [
            {
                "start": "00:00:02.389",
                "end": "00:00:02.399",
                "lines": [
                    "hello and welcome if you want to study"
                ]
            },
            {
                "start": "00:00:04.280",
                "end": "00:00:04.290",
                "lines": [
                    "different phasma learning don't go"
                ]
            },
            ...
]
```
