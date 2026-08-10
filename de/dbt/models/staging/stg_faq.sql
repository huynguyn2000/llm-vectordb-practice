select
    id,
    trim(question) as question,
    trim(answer) as answer,
    lower(topic) as topic
from {{ ref('faq') }}
