select
    id,
    'faq:' || id as source,
    question || chr(10) || chr(10) || answer as content
from {{ ref('stg_faq') }}
