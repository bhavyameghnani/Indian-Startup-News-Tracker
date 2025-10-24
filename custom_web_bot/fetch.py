import sqlite3

def connect_db():
    return sqlite3.connect('news.db')

def get_pos_by_tag_and_subtag(tag_names=None, subtag_names=None):
    if not tag_names and not subtag_names:
        print("Error: At least one of tag_names or subtag_names must be provided.")
        return []

    conn = connect_db()
    cursor = conn.cursor()

    conditions = []
    params = []

    # Build tag filtering condition for intersection (ALL tags must match)
    if tag_names:
        tag_conditions = []
        for tag in tag_names:
            tag_conditions.append(
                "a.id IN ("
                "SELECT at.article_id "
                "FROM article_tags at "
                "JOIN tags t ON at.tag_id = t.id "
                "WHERE t.name = ?)"
            )
            params.append(tag)
        # Join all tag conditions with AND to get intersection
        tag_condition = ' AND '.join(tag_conditions)
        conditions.append(tag_condition)

    # Build subtag filtering condition if provided (ANY subtag can match)
    if subtag_names:
        subtag_placeholders = ','.join(['?' for _ in subtag_names])
        subtag_condition = (
            "a.id IN ("
            "SELECT ast.article_id "
            "FROM article_subtags ast "
            "JOIN subtags st ON ast.subtag_id = st.id "
            f"WHERE st.name IN ({subtag_placeholders}))"
        )
        conditions.append(subtag_condition)
        params.extend(subtag_names)

    # Combine conditions with AND logic
    where_clause = ' AND '.join(conditions)
    query = (
        "SELECT DISTINCT a.pos "
        "FROM articles a "
        f"WHERE {where_clause} AND a.pos IS NOT NULL;"
    )

    # Debug
    print("Executing Query:")
    print(query)
    print("With parameters:", params)

    cursor.execute(query, params)
    results = cursor.fetchall()

    pos_list = [row[0] for row in results]

    cursor.close()
    conn.close()

    print(f"Found {len(pos_list)} matching articles.")
    return pos_list
