Question to discuss with Ryan:
- Aren't the commuators for the gradient an approximation? The generators don't commute with the unitaries.
- How will we do matrix exponentials in the TN setting? Will we just compile a circuit? See line 2831 of `adapt_vqe.py`.
- Line 2887 of `adapt_vqe.py` is missing the thing that we return!
- Line 3099: Is there a way to conjuate an MPO with a unitary in quimb?
- Line 3108: Check in debugger if `left_matrix` is actually a (state) vector.
- Line 2857: How do we do this? Would it be the same as multiplying the exp by a vector?