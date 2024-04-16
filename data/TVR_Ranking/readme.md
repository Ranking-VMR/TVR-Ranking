
## TVR-Ranking Dataset Introduction
The TVR-Ranking dataset is built based on [TVR](https://github.com/jayleicn/TVRetrieval) to support the Video Moment Retrieval Ranking task.
Given a query, each moment comes with a five-level relevance in our dataset.
We provide the ground truth of validation and test set. We also generate the top 1, top 20, and top 40 pseudo training set based on query-caption similarity.
The raw annotation was released to encourage further exploration.

The full dataset file can be downloaded from [TVR-Ranking](
https://drive.google.com/drive/folders/1QuE3Ah1VR_Sudjbl_5VFC1J-aT9Dh_WF?usp=drive_link).

## Dataset Information


### Data Organization 
Here we show how the data is organized.
```
TVR_Ranking/
  -val.jsonl                  
  -test.jsonl                 
  -train_top01.jsonl
  -train_top20.jsonl
  -train_top40.jsonl
  -video_name_duration_id.json          # a map including video name, duration, video id 

  -raw_query_moment_pair.csv            # the basic information for query-moment pair
  -raw_annotation.csv                   # all the raw annotation 

```

The moments in raw annotation were labeled by 2 or 4 workers. You could use merger_raw_annotation.py to merge them into one file. In the validation/test set, we remove the unconsensus annotation and average the relevances, so there is only one label for each moment 



### Data Formats

The TVR-Ranking dataset contains the following information:
``` 
pair_id: 		  A unique identifier for the query-moment pair.
query_id: 	  A unique identifier for the query.
query: 		    The textual sentence that the user may want to know.
video_name: 	The name of the video file.
timestamp: 	  The start and end times (in seconds) to identify a moment.
duration: 	  The duration of the video (in seconds).
caption: 	    A textual description of the content of the moment.
similarity: 	A score (0 to 1) indicates the similarity between the query and the caption of the moment.
worker: 		  The unique identifier of each worker.
relevance:  	A rating (0 to 4) indicating the relevance of the moment to the query, with 4 being the most relevant.
```


An example from validation was shown as follows.
``` json
{
  "pair_id": 0,
  "query_id": 54251,
  "query": "A man and a woman are talking about how he lied to a patient.",
  "video_name": "house_s07e03_seg02_clip_24",
  "timestamp": [63.47, 77.42],
  "duration": 90.02,
  "caption": "A man and a woman are talking about how he lied to a patient.",
  "similarity": 1.0,
  "relevance": 4
}
```






## License
[Specify the license under which the TVR-Ranking dataset is distributed. Mention any restrictions, permissions, and conditions imposed by the license.]

## Citing the Dataset
If you use the TVR-Ranking dataset in your research, please cite it as follows:

```
citation
```

## Contact Information
For any questions or further information regarding the TVR-Ranking dataset, please contact Renjie Liang at liangrj5@gmail.com.
