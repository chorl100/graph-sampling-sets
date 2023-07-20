CATEGORICAL_COLUMNS = ["age_segment", "federal_state", "region_type", "emp_liable", "mixed_hh_flag", "orig_sf",
                       "serv_sf", "hou_fam_structure", "hou_aff_new_products", "hou_aff_prices", "provider_desc",
                       "network_desc", "subs_hand_ind", "bnt_vvl_lng", "current_tariff_option", "l3_tariff_desc",
                       "l4_tariff_desc", "l5_tariff_desc", "tariff_group", "article_status_enc", "tech_generation",
                       "device_type", "producer_sold", "operating_system", "producer_used", "handset_feat_type",
                       "known_article", "equal_to_recently_sold"]

PROTECTED_COLUMNS = ['client_id', 'label', 'nbr_campaign_responded', 'nbr_campaign_taken']

VARIABLES_TO_EXCLUDE = [
    # The following variables are not useful predictions, because they are keys, or contain too many levels
    'canc_flag',
    'pseudo_id',
    'slot_month',
    'adr_zip',
    'current_tariff_option',
    'l3_tariff_desc',
    'l4_tariff_desc',
    'l5_tariff_desc',
    'avg_nps_tot_3m'
    'avg_nps_tot_6m',
    'avg_nps_tot_12m',
    'wavg_nps_tot',
    'avg_nps_nps_3m'
    'avg_nps_nps_6m',
    'avg_nps_nps_12m',
    'wavg_nps_nps'
    # The following variables provide a glimpse at the future, and therefore need to be removed
    'provider_desc',
    'network_desc',
    'cost_delta',
    'cost_delta_pct',
    'has_kd_products',
    'mixed_hh_flag',
    'adv_permission',
    # Variables that only contain NULLs
    'tot_nbr_calls',
    'nbr_dropped_calls',
    'nbr_unsuc_call_attempts',
    'max_fd_call_drops',
    'max_fd_unsuc_call_attempts',
    'pct_dropped_calls',
    'pct_call_success',
    'avg_bandwidth_in_avg',
    'avg_bandwidth_in_std',
    'avg_bandwidth_in_trnd6m',
    'avg_bandwidth_out_avg',
    'avg_bandwidth_out_std',
    'avg_bandwidth_out_trnd6m',
    'avg_days_of_usage_avg',
    'avg_days_of_usage_std',
    'avg_days_of_usage_trnd6m',
    'nbr_sessions_avg',
    'nbr_sessions_std',
    'nbr_sessions_trnd6m',
    'sum_downlink_dvolume_avg',
    'sum_downlink_dvolume_std',
    'sum_downlink_dvolume_trnd6m',
    'sum_duration_avg',
    'sum_duration_std',
    'sum_duration_trnd6m',
    'sum_uplink_dvolume_avg',
    'sum_uplink_dvolume_std',
    'sum_uplink_dvolume_trnd6m',
    # Other
    'remaining_days',
    'run_id',
    'partition_year_month_day'
]

TEXTUAL_COLUMNS = [
    'federal_state',
    'orig_sf',
    'serv_sf',
    'provider_desc',
    'tariff_group',
    'device_type',
    'producer_sold',
    'operating_system',
    'producer_used',
    'handset_feat_type'
]

RELEVANT_COLUMNS = [
    'adr_zip', 'avg_canc_adr_zip', 'avg_canc_age_segment',
    'avg_canc_federal_state', 'avg_canc_orig_sf', 'avg_canc_serv_sf',
    'avg_cost', 'avg_nps_nps_3m', 'avg_nps_tot_3m',
    'avg_subs_excl_inst', 'avg_subs_incl_inst', 'avg_use_time',
    'client_id', 'cost',
    'data_number_wap_avg', 'data_number_wap_std',
    'data_number_wap_trnd3m', 'data_number_wap_trnd6m',
    'data_vol_roaming_avg', 'data_vol_roaming_std',
    'data_vol_roaming_trnd3m', 'data_vol_roaming_trnd6m',
    'data_vol_total_avg', 'data_vol_total_std',
    'data_vol_total_trnd3m', 'data_vol_total_trnd6m',
    'days_since_last_use_sold', 'days_since_latest_use',
    'discount_avg', 'discount_std', 'discount_trnd3m',
    'discount_trnd6m', 'ext_credit_class', 'gender',
    'gross_nbr_handsets_sold', 'instalment_avg', 'instalment_std',
    'instalment_trnd3m', 'instalment_trnd6m', 'int_credit_class',
    'mon_since_con_start', 'mon_since_last_subsi',
    'mon_til_next_subsi', 'months_since_act', 'nbr_campaigns',
    'nbr_canc_req', 'nbr_cold_ho', 'nbr_complaints_12m',
    'nbr_complaints_3m', 'nbr_complaints_6m', 'nbr_con_cycles',
    'nbr_consult', 'nbr_cust_facing_rebook', 'nbr_distinct_teams',
    'nbr_handsets_returned', 'nbr_handsets_used', 'nbr_past_contracts',
    'nbr_past_subsi_dates', 'nbr_past_tariffs',
    'nbr_purchase_trans_avg', 'nbr_purchase_trans_max',
    'nbr_purchase_trans_std', 'nbr_purchase_trans_trnd12m',
    'nbr_return_debit_notes', 'nbr_upset_comp_6m', 'nbr_warm_ho',
    'nbr_winback_accept', 'pct_campaign_responded',
    'pct_campaign_taken', 'pct_winback_accept', 'qu_avg_hou_pp',
    'qu_avg_liv_area', 'qu_gini_hou_pp', 'qu_gini_liv_area',
    'qu_inhab_per_sqkm', 'qu_nbr_inhabitants', 'qu_pct_foreigners',
    'qu_pct_hh_acad_title', 'qu_pct_hh_foreign', 'qu_pp_inhabitants',
    'qu_std_hou_pp', 'qu_std_liv_area', 'quantity', 'remaining_months',
    'revenue_avg', 'revenue_std', 'revenue_trnd3m', 'revenue_trnd6m',
    'serv_revenue_avg', 'serv_revenue_std', 'serv_revenue_trnd3m',
    'serv_revenue_trnd6m', 'sold_in_month', 'str_avg_hou_pp',
    'str_avg_inhab_hh', 'str_avg_liv_area', 'str_gini_hou_pp',
    'str_gini_liv_area', 'str_nbr_households', 'str_pct_hh_acad_title',
    'str_pct_hh_foreign', 'str_std_hou_pp', 'str_std_liv_area',
    'subs_excl_inst', 'subs_incl_inst', 'sum_net_amount_avg',
    'sum_net_amount_max', 'sum_net_amount_std',
    'sum_net_amount_trnd12m', 'time_first_queue_wait',
    'time_first_ring_wait', 'time_hold', 'time_talk',
    'tot_ivr_duration', 'tot_nbr_complaints', 'tot_nbr_hotl_calls',
    'tot_nbr_trans_avg', 'tot_nbr_trans_max', 'tot_nbr_trans_std',
    'tot_nbr_trans_trnd12m', 'tot_payment_issues', 'vodafone_stars',
    'voice_min_international_avg', 'voice_min_international_std',
    'voice_min_international_trnd3m', 'voice_min_international_trnd6m',
    'voice_min_offnet_avg', 'voice_min_offnet_std',
    'voice_min_offnet_trnd3m', 'voice_min_offnet_trnd6m',
    'voice_min_onnet_avg', 'voice_min_onnet_std',
    'voice_min_onnet_trnd3m', 'voice_min_onnet_trnd6m',
    'voice_min_roaming_avg', 'voice_min_roaming_std',
    'voice_min_roaming_trnd3m', 'voice_min_roaming_trnd6m',
    'voice_min_wireline_avg', 'voice_min_wireline_std',
    'voice_min_wireline_trnd3m', 'voice_min_wireline_trnd6m',
    'voucher_avg', 'voucher_std', 'voucher_trnd3m', 'voucher_trnd6m',
    'year_of_birth', 'zip_inhab_per_sqkm', 'zip_nbr_inhabitants',
    'zip_pct_foreigners', 'zip_pct_hh_acad_title',
    'zip_pct_hh_foreign', 'zip_pp_inhabitants'
]