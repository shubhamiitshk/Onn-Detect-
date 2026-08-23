package com.onndetect.ui.navigation

import androidx.compose.runtime.Composable
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.navArgument
import com.onndetect.ui.screens.history.HistoryScreen
import com.onndetect.ui.screens.home.HomeScreen
import com.onndetect.ui.screens.result.ResultScreen
import com.onndetect.ui.screens.scanner.ScannerScreen

object Routes {
    const val HOME = "home"
    const val SCANNER = "scanner?uri={uri}&mode={mode}"
    const val RESULT = "result/{scanId}"
    const val HISTORY = "history"
}

@Composable
fun ONNNavGraph(navController: NavHostController) {
    NavHost(navController, startDestination = Routes.HOME) {
        composable(Routes.HOME) { HomeScreen(navController) }

        composable(
            route = Routes.SCANNER,
            arguments = listOf(
                navArgument("uri") {
                    type = NavType.StringType
                    defaultValue = ""
                },
                navArgument("mode") {
                    type = NavType.StringType
                    defaultValue = "IMAGE_ONLY"
                },
            ),
        ) { backStackEntry ->
            ScannerScreen(
                navController = navController,
                uri = backStackEntry.arguments?.getString("uri")?.takeUnless { it.isEmpty() },
                mode = backStackEntry.arguments?.getString("mode"),
            )
        }

        composable(
            route = Routes.RESULT,
            arguments = listOf(navArgument("scanId") { type = NavType.StringType }),
        ) { backStackEntry ->
            ResultScreen(
                navController = navController,
                scanId = backStackEntry.arguments?.getString("scanId"),
            )
        }

        composable(Routes.HISTORY) { HistoryScreen(navController) }
    }
}
