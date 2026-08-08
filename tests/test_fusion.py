from search.fusion import reciprocal_rank_fusion


def test_empty_input_returns_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_single_list_preserves_order():
    fused = reciprocal_rank_fusion([[3, 1, 2]])
    assert [doc_id for doc_id, _ in fused] == [3, 1, 2]


def test_single_list_order_independent_of_k():
    order_small = [i for i, _ in reciprocal_rank_fusion([[3, 1, 2]], k=1)]
    order_large = [i for i, _ in reciprocal_rank_fusion([[3, 1, 2]], k=1000)]
    assert order_small == order_large == [3, 1, 2]


def test_id_in_both_lists_outranks_id_in_one():
    # doc 2 appears in both lists; doc 1 only in the first at rank 0.
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 4, 5]])
    order = [doc_id for doc_id, _ in fused]
    assert order[0] == 2            # in both -> highest fused score
    assert order.index(1) < order.index(4)  # rank-0 single beats rank-1 single


def test_all_ids_present_as_union():
    fused = reciprocal_rank_fusion([[1, 2], [2, 3]])
    assert {doc_id for doc_id, _ in fused} == {1, 2, 3}


def test_score_formula_zero_based_rank():
    # single doc at rank 0 -> 1/(k+0)
    fused = reciprocal_rank_fusion([[7]], k=60)
    assert fused == [(7, 1.0 / 60)]


def test_tie_break_is_ascending_id():
    # doc 1 and doc 2 each appear once at rank 0 -> equal score -> id order
    fused = reciprocal_rank_fusion([[1], [2]])
    assert [doc_id for doc_id, _ in fused] == [1, 2]
