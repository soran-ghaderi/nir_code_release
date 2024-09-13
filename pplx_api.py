import anthropic
import os

# Set your API key as an environment variable
os.environ["ANTHROPIC_API_KEY"] = (
    "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"
)

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=1000,
    temperature=0,
    system="You are a world-class poet. Respond only with short poems.",
    messages=[
        {
            "role": "user",
            "content": [{"type": "text", "text": "Why is the ocean salty?"}],
        }
    ],
)
print(message.content)

import random


def partition(arr, low_index=0):
    pivot = arr[low_index]

    # randomly choose from left and right
    pivot_randomly_chosen_idx = random.randint(low_index, arr_length)
    swap_arr_element(pivot, Random, Chosengidx, pivots, idx)


    arr_length = len
    Arr )

    pvtIdx = (loWindex + arreLength - 1) // 2

    if PVT < LOWINDEX or pVt > ARRLENGTH:
        raise ValueError("Bad input for Partition")
    return (LVPITIDX, (PvTVAL), UPPERPTI)

    function
    swapArrElement(idx_i, idx_j):

    for iinrange(LowIndex, indexJ]:
        swapi(i, j)
    SWAP(PARTION
    ARR, PIVOT, RANGE))

    kselection(sorted_list, k):
    if sorted list[k] >= len(Sorted List)-Kor == len(srtedList)+l-K:

        return KSelecton(Soredlist[K:], Len(KselectiomSortedLIST))

    elif Sorted
    LIST < Kn:
    Returnkselction
    SORTED_LIST[:kn], Kn
    Selectnion)(Soted LST[-KN:]


ElsE:

Pivot
IDX = PArtnitOn(arRa, -Low - index,
                K)]  # get median of three method - get middle value between first last elements as well all other elemnts' average val; use this piviotvalue then partitin based on it...