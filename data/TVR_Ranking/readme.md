
## TVR-Ranking Dataset Introduction
We curated a TVR-Ranking dataset to support the Ranked Video Moment Retrieval (RVMR) task, the videos and queries are sourced from [TVR](https://github.com/jayleicn/TVRetrieval).
Our dataset evaluates models based on their ability to retrieve relevant video moments. Annotators scored relevance on a five-level scale for each moment.

The validation and test sets were manually annotated. Additionally, we generated pseudo training sets (top 1, top 20, and top 40) based on query-caption similarity. Raw annotations are released to encourage further exploration. Video durations were aligned by frame number, which may result in slight differences compared to the TVR dataset. The full dataset file can be downloaded from Hugging Face at [TVR-Ranking](https://huggingface.co/axgroup/TVR-Ranking).



### Data Organization 
Here we show how the data is organized.
```
TVR_Ranking/
  -val.json                  
  -test.json                 
  -train_top01.json
  -train_top20.json
  -train_top40.json
  -raw_annotation.csv
```

The moments in raw annotation were labeled by 2 or 4 workers and we remain the unconsence data. In the validation/test set, we remove the unconsensus annotation and average the relevances.

### Data Formats

The TVR-Ranking dataset contains the following information:
``` 
pair_id:      A unique identifier for the query-moment pair.
query_id:     A unique identifier for the query.
query:        The textual sentence that the user may want to know.
video_name:   The name of the video file.
timestamp:    The start and end times (in seconds) to identify a moment.
duration:     The duration of the video (in seconds).
caption:      A textual description of the content of the moment.
similarity:   A score (-1 to 1) indicates the similarity between the query and the caption of the moment.
worker:       The unique identifier of each worker.
relevance:    A rating (0 to 4) indicating the relevance of the moment to the query, with 4 being the most relevant.
```


An example from validation was shown as follows.
``` json
{
  "pair_id":      0,
  "query_id":     54251,
  "query":        "A man and a woman are talking about how he lied to a patient.",
  "video_name":   "house_s07e03_seg02_clip_24",
  "timestamp":    [63.47, 77.42],
  "duration":     90.02,
  "caption":      "A man and a woman are talking about how he lied to a patient.",
  "similarity":   1.0,
  "relevance":    4
}
```


## License
This project and dataset are licensed under a Creative Commons license.


