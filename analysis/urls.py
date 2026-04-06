from django.urls import path
from .views import (root_analysis_view, analyze_stream_view,
                    download_all_xlsx, download_xlsx,
                    sam_prompt_view, reanalyze_view,
                    merge_masks_view, download_overlays_view,
                    export_training_view, export_training_batch_view)

urlpatterns = [
    path('', root_analysis_view, name='analysis'),
    path('analyze_stream/', analyze_stream_view, name='analyze_stream'),
    path('download_xlsx/', download_xlsx, name='download_xlsx'),
    path('download_all_xlsx/', download_all_xlsx, name='download_all_xlsx'),
    path('sam_prompt/', sam_prompt_view, name='sam_prompt'),
    path('reanalyze/', reanalyze_view, name='reanalyze'),
    path('merge_masks/', merge_masks_view, name='merge_masks'),
    path('download_overlays/', download_overlays_view, name='download_overlays'),
    path('export_training/', export_training_view, name='export_training'),
    path('export_training_batch/', export_training_batch_view, name='export_training_batch'),
]