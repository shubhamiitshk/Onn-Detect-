package com.onndetect.ui.navigation

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import android.net.Uri
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.navArgument
import com.onndetect.ui.screens.history.HistoryScreen
import com.onndetect.ui.screens.home.HomeScreen
import com.onndetect.ui.screens.result.ResultScreen
import com.onndetect.ui.screens.scanner.ScannerScreen
import com.onndetect.ui.share.SharedMedia

object Routes {
    const val HOME = "home"
    const val SCANNER = "scanner?uri={uri}&mode={mode}"
    const val RESULT = "result/{scanId}"
    const val HISTORY = "history"

    /** The scanner route for a URI, with the URI encoded for the route. */
    fun scanner(uri: String, mode: String): String =
        "scanner?uri=${Uri.encode(uri)}&mode=${Uri.encode(mode)}"
}

@Composable
fun ONNNavGraph(navController: NavHostController, sharedMedia: SharedMedia? = null) {
    // A share hand-off starts the scan immediately, but HOME stays
    // UNDERNEATH it in the back stack, so the back button returns to the
    // home screen rather than leaving the app.
    LaunchedEffect(sharedMedia) {
        val shared = sharedMedia ?: return@LaunchedEffect
        navController.navigate(Routes.scanner(shared.uri, shared.mode))
    }
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
            )
        }

        composable(Routes.HISTORY) { HistoryScreen(navController) }

    }
}
