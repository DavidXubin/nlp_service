Q: How to start the sentiment service:

A: In sentiment_service directory:

1. celery multi start sentiment --app=sentiment:celery_app -B --loglevel=info --logfile=logs/sentiment_celery_task.log --pidfile=sentiment_celery_task.pid

2. nohup python web_service.py >> logs/web_service.log &

3. nohup python web_service_topic.py >> logs/web_service_topic.log &

4. nohup python web_service_relation_check.py >> logs/web_service_relation_check.log &

5. nohup python sentiment_save_data_service.py >/dev/null 2>&1 &

6. nohup python sentiment_update_service.py >/dev/null 2>&1 &

7. Add below command into crontab:

3 23 * * * cd /data/workspace/jupyter_env/sentiment/sentiment_service && /data/workspace/jupyter_env/bin/python /data/workspace/jupyter_env/sentiment/sentiment_service/sentiment_commit_model.py >/dev/null 2>&

Please wait about 20 minutes for the service startup and it is very slow to load word vector model

8. celery multi start topic_analysis --app=topic_analysis:celery_app -B --loglevel=info --logfile=logs/topic_celery_task.log --pidfile=topic_celery_task.pid

9. nohup python topic_analysis_service.py >/dev/null 2>&1 &

10. nohup python pyspark_topic_analysis.py >/dev/null 2>&1 &

11. nohup python ner_predict_service.py >/dev/null 2>&1 &
Note: ner_predict_service.py should be launched fourthly as 4 processes

12. nohup python dedup_service.py >/dev/null 2>&1 &

13. nohup python text_summary_extract.py >/dev/null 2>&1 &

Q: How to stop the sentiment service:

1. celery multi stop sentiment --pidfile=sentiment_celery_task.pid

2. celery multi stop topic_analysis --pidfile=topic_celery_task.pid

3. Use 'ps aux | grep sentiment' to find the sentiment processes pid 

4. Use 'kill -9 <sentiment process pid>' to kill the processes

5. Use 'ps aux | grep topic_analysis' to find the topic analysis processes pid

6. Use 'kill -9 <topic_analysis process pid>' to kill the processes
