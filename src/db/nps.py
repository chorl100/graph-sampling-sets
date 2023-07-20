import src.db.big_query as bq


def nps_signal(from_date: str, to_date: str, client_ids):
    """
    Generates a signal vector from NPS recommendation scores in a given time frame.
    Each entry is in {0, 1, 2, 3, ..., 10}.
    :param from_date: start date
    :param to_date: end date
    :param client_ids: list-like client ids whose scores to include
    :return: NPS signal vector
    """
    print("Reading NPS recommendation values...")
    answers_df = bq.nps_query_timeframe(from_date, to_date).astype(int)
    # keep only those answer values that can be assigned to known customers
    answers_df = answers_df[answers_df.client_id.isin(client_ids)]
    s = answers_df.answer_value.to_numpy().astype(int)
    return s
