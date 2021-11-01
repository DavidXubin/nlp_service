import os
import re
import sys
import datetime
import json
import pandas as pd
from multiprocessing import Process
from datetime import timedelta
import config
from chinese_ner.utils import check_related_company_v2

from utils import get_logger

logger = None
if logger is None:
    logger = get_logger("clean-company-data")


def _preprocess_real_estate_company_data_internal(data_path, start_date, end_date):

    try:
        if not os.path.exists(data_path):
            raise Exception("{} does not exist".format(data_path))

        logger.info("Start to preprocess real estate company topic data from {} to {}".format(start_date, end_date))

        df = pd.read_csv(data_path, sep=',')
        logger.info("Real estate company df shape: {}".format(df.shape))

        company_df = df[(df["pub_date"] >= start_date) & (df["pub_date"] <= end_date)]

        related_indices = []
        for idx in company_df.index:
            content = company_df.loc[idx, "content"]
            if pd.isnull(content) or len(content.strip()) == 0:
                continue

            products = company_df.loc[idx, "products"]
            if pd.isnull(products):
                continue

            if products.find("[") >= 0 and products.find("]") > 0:
                products = json.loads(products)
            else:
                products = re.split("[,，]", products)

            company_names = [product for product in products if product.strip() != config.CompanyType.REAL_ESTATE.value]
            if len(company_names) > 0:
                related = check_related_company_v2(content, company_names[0].strip())

                if not related:
                    logger.info("{}: {} has unrelated content".format(company_names[0], company_df.loc[idx, "crawler_id"]))
                    continue

            related_indices.append(idx)

        logger.info("Negative text number is {} between {} and {}".format(len(related_indices), start_date, end_date))

        if len(related_indices) == 0:
            return

        company_df = company_df.loc[related_indices, :]
        company_df.to_csv(os.path.join(config.topic_data_tmp_path, config.CompanyType.REAL_ESTATE.value + "_" + start_date + ".csv"),
                          sep=',', header=True, index=False)

        logger.info("Preprocessed real estate company df shape is {} between {} and {}".format(company_df.shape, start_date, end_date))
    except Exception as e:
        logger.error(e)


def _preprocess_real_estate_company_data(filepath, start_date):
    if not os.path.exists(config.topic_data_tmp_path):
        os.makedirs(config.topic_data_tmp_path)

    if not os.path.exists(filepath):
        raise Exception("{} does not exist".format(filepath))

    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    logger.info("start date is {}".format(start_date))
    logger.info("end date is {}".format(end_date))

    date_range = pd.date_range(start=start_date, end=end_date, freq='W').tolist()
    if pd.Timestamp(start_date) < date_range[0]:
        date_range.insert(0, pd.Timestamp(start_date))

    if pd.Timestamp(end_date) > date_range[-1]:
        date_range.append(pd.Timestamp(end_date))

    date_range = [pd.to_datetime(x) for x in date_range]
    logger.info(date_range)

    df_paths = []
    _processes = []
    for i, date in enumerate(date_range):
        if i == len(date_range) - 1:
            break

        start = date_range[i]
        if i > 0:
            start = start + timedelta(days=1)
        end = date_range[i + 1]

        start = start.strftime("%Y-%m-%d")
        end = end.strftime("%Y-%m-%d")

        logger.info("Process[{}]: start date={}, end date={}".format(i, start, end))

        p = Process(target=_preprocess_real_estate_company_data_internal, args=(filepath, start, end,))
        p.start()
        _processes.append(p)

        df_paths.append(os.path.join(config.topic_data_tmp_path, config.CompanyType.REAL_ESTATE.value + "_" + start + ".csv"))

    for p in _processes:
        p.join()

    all_df = None
    for path in df_paths:
        if not os.path.exists(path):
            logger.error("{} does not exists !".format(path))
            continue

        df = pd.read_csv(path,  sep=',')
        if all_df is None:
            all_df = df
        else:
            all_df = pd.concat([all_df, df], axis=0)

        os.remove(path)

    preprocessed_real_estate_data_path = os.path.join(config.topic_data_tmp_path, "preprocessed_texts.csv")

    if all_df is not None and all_df.shape[0] > 0:
        all_df.to_csv(preprocessed_real_estate_data_path, sep=',', header=True, index=False)


def _execute(filepath, start_date):
    _preprocess_real_estate_company_data(filepath, start_date)


if __name__ == "__main__":
    _execute(sys.argv[1], sys.argv[2])
