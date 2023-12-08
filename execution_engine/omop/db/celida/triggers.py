trigger_interval_overlap_check_function_sql = """
CREATE OR REPLACE FUNCTION check_overlapping_intervals()
RETURNS TRIGGER AS $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM {schema}.{table}
        WHERE NEW.person_id = person_id
        AND NEW.recommendation_run_id = recommendation_run_id
        AND NEW.plan_id IS NOT DISTINCT FROM plan_id
        AND NEW.criterion_id IS NOT DISTINCT FROM criterion_id
        AND NEW.cohort_category IS NOT DISTINCT FROM cohort_category
        AND (NEW.interval_start, NEW.interval_end) OVERLAPS (interval_start, interval_end)
    ) THEN
        RAISE EXCEPTION 'Overlapping intervals detected';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

"""
create_trigger_interval_overlap_check_sql = """
CREATE TRIGGER trigger_recommendation_result_interval_before_insert
BEFORE INSERT ON {schema}.{table}
FOR EACH ROW EXECUTE FUNCTION check_overlapping_intervals();
"""
