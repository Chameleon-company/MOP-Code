import type { RefObject } from '@mui/x-internals/types';
import type { GridDataSourceCache } from "../../../models/gridDataSource.js";
import { CacheChunkManager, DataSourceRowsUpdateStrategy } from "./utils.js";
import type { GridDataSourceApi, GridDataSourceBaseOptions } from "./models.js";
import type { GridPrivateApiCommunity } from "../../../models/api/gridApiCommunity.js";
import type { DataGridProcessedProps } from "../../../models/props/DataGridProps.js";
import type { GridStrategyProcessor } from "../../core/strategyProcessing/index.js";
import type { GridEventListener } from "../../../models/events/index.js";
import type { GridRowId } from "../../../models/gridRows.js";
export declare const useGridDataSourceBase: <Api extends GridPrivateApiCommunity>(apiRef: RefObject<Api>, props: Pick<DataGridProcessedProps, "dataSource" | "dataSourceCache" | "onDataSourceError" | "pageSizeOptions" | "pagination" | "signature" | "dataSourceRevalidateMs">, options?: GridDataSourceBaseOptions) => {
  api: {
    public: GridDataSourceApi;
  };
  debouncedFetchRows: ((parentId?: GridRowId, params?: import("./models.js").GridDataSourceFetchRowsParams<import("@mui/x-data-grid").GridGetRowsParams>) => Promise<void>) & import("@mui/utils/debounce").Cancelable;
  strategyProcessor: {
    strategyName: DataSourceRowsUpdateStrategy;
    group: "dataSourceRowsUpdate";
    processor: GridStrategyProcessor<"dataSourceRowsUpdate">;
  };
  setStrategyAvailability: () => void;
  startPolling: () => void;
  stopPolling: () => void;
  cacheChunkManager: CacheChunkManager;
  cache: GridDataSourceCache;
  events: {
    strategyAvailabilityChange: GridEventListener<"strategyAvailabilityChange">;
    sortModelChange: (...params: unknown[]) => void;
    filterModelChange: (...params: unknown[]) => void;
    paginationModelChange: (...params: unknown[]) => void;
  };
};